# SVAR2 Contig-Name Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every place `SparseVar2` matches a caller-supplied contig name robust to `chr`-prefix and mito-alias naming differences (`10` ↔ `chr10`), consistent with `ContigNormalizer`.

**Architecture:** Two independent parts. **Part A (Python)** adds a `ContigNormalizer` to `SparseVar2` and routes reader-side contig arguments through resolve helpers. **Part B (Rust)** resolves the query contig against the reference FASTA's own naming at the single `vcf_reader.rs` choke point. Both mirror `ContigNormalizer`'s equivalence rule.

**Tech Stack:** Python (pytest, polars, numpy), Rust (rust-htslib faidx, PyO3), maturin, pixi.

**Spec:** `docs/superpowers/specs/2026-07-19-svar2-contig-normalization-design.md`

## Global Constraints

- **Conventional Commits** for every commit (`feat:`, `fix:`, `test:`, `docs:`, `perf:`).
- **Never edit `CHANGELOG.md`** or bump the version by hand — CI/commitizen owns them.
- **Public-API docs:** any change to a public name/behavior requires updating `skills/genoray-api/SKILL.md` in the same PR (Task 8).
- **Rust tests need `--no-default-features`**: run as `pixi run bash -lc 'cargo test --no-default-features <args>'`, else the PyO3 test binary fails to link (`undefined symbol: _Py_Dealloc`). The contig resolver lives behind the `conversion` feature (it's in `vcf_reader.rs`); build/test it with `--features conversion` **and** run a `cargo check --no-default-features` on `lib.rs` gating if module visibility changes.
- **NFS `target/` breaks cargo** in this worktree: export a non-NFS target dir for all `cargo`/pre-commit-hook runs — `export CARGO_TARGET_DIR=/tmp/genoray-target-$USER` (or under `$CLAUDE_JOB_DIR/tmp`) before `cargo test`/`git commit`.
- **`pixi run test` does NOT rebuild the Rust `.so`.** After any Rust change, run `pixi run maturin develop --release` before Python-level verification of that change.
- Coordinate convention: 0-based, half-open `[start, end)`. Contigs in the test fixtures are `chr`-prefixed (`_REF`/`svar2_store` in `tests/conftest.py`).

---

## Parallelization

- **Wave 1 (parallel):** Task 1 (Python helpers + reader methods) ∥ Task 4 (Python parity table) ∥ Task 5 (Rust resolver). Task 4 depends only on `ContigNormalizer` (already in the tree), so it can run immediately.
- **Wave 2 (parallel):** Task 2 ∥ Task 3 (both depend on Task 1).
- **Wave 3:** Task 6 (rebuild `.so` + Python e2e, depends on Task 5), then Task 7 (docs, depends on all).

Task 4's Python parity table and Task 5's Rust table must stay byte-identical; if the reviewer edits one, update the other in the same pass.

Dispatch waves with `superpowers:dispatching-parallel-agents` + `superpowers:subagent-driven-development`; review between tasks.

---

## Task 1: Python resolve helpers + reader query methods

**Files:**
- Modify: `python/genoray/_svar2.py` (`SparseVar2.__init__` ~line 345-350; add helper methods after `__init__`)
- Modify: `python/genoray/_svar2_decode.py` (`decode`, `region_counts`)
- Modify: `python/genoray/_svar2_batch.py` (`_overlap_batch`, `read_ranges`, `_find_ranges`, `_gather_ranges`)
- Test: `tests/test_svar2_contig_alias.py` (new)

**Interfaces:**
- Produces: `SparseVar2._cnorm: ContigNormalizer`; `SparseVar2._resolve_contig(self, contig: str) -> str` (raises `ValueError` on miss); `SparseVar2._resolve_contigs(self, contigs: Sequence[str]) -> list[str]`; `SparseVar2._reader(self, contig: str) -> _core.PyContigReader`. Consumed by Tasks 2, 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_svar2_contig_alias.py`:

```python
"""Reader-side contig-name normalization for SparseVar2 (chr-prefix + mito aliases)."""

import numpy as np
import pytest

from genoray import SparseVar2


def test_decode_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)  # store contig is "chr1"
    native = sv.decode("chr1", [(0, 40)])
    alias = sv.decode("1", [(0, 40)])
    assert native["pos"].lengths.reshape(-1).tolist() == alias["pos"].lengths.reshape(-1).tolist()
    assert np.array_equal(np.asarray(native["pos"].data), np.asarray(alias["pos"].data))


def test_region_counts_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    assert np.array_equal(
        sv.region_counts("chr1", [(0, 40)]), sv.region_counts("1", [(0, 40)])
    )


def test_read_ranges_accepts_unprefixed_contig(svar2_store):
    sv = SparseVar2(svar2_store)
    native = sv.read_ranges("chr1", [0], [40])
    alias = sv.read_ranges("1", [0], [40])
    assert np.array_equal(native["vk_pos"], alias["vk_pos"])
    assert np.array_equal(native["vk_key"], alias["vk_key"])


def test_unknown_contig_raises_valueerror(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in store"):
        sv.decode("chrZ", [(0, 40)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_contig_alias.py -v`
Expected: FAIL — `sv.decode("1", ...)` raises `KeyError: '1'` (no normalization yet).

- [ ] **Step 3: Add `_cnorm` + resolve helpers to `SparseVar2`**

In `python/genoray/_svar2.py`, at the end of `__init__` (after `self._readers = {...}`):

```python
        from genoray._contigs import ContigNormalizer

        self._cnorm = ContigNormalizer(self.contigs)
```

Immediately after `__init__` (before `n_samples`), add:

```python
    def _resolve_contig(self, contig: str) -> str:
        """Resolve a caller-supplied contig name to the store's own spelling.

        Handles ``chr``-prefix and mito-alias differences via the store's
        :class:`~genoray._contigs.ContigNormalizer` (e.g. ``"1"`` -> ``"chr1"``).
        Raises ``ValueError`` if no store contig is equivalent.
        """
        norm = self._cnorm.norm(contig)
        if norm is None:
            raise ValueError(
                f"Contig {contig!r} not found in store; available contigs: {self.contigs}"
            )
        return norm

    def _resolve_contigs(self, contigs: "Sequence[str]") -> list[str]:
        """Resolve each contig via :meth:`_resolve_contig`, preserving order."""
        return [self._resolve_contig(c) for c in contigs]

    def _reader(self, contig: str) -> "_core.PyContigReader":
        """The per-contig Rust reader for ``contig``, resolving naming schemes."""
        return self._readers[self._resolve_contig(contig)]
```

(`Sequence` and `_core` are already imported in this module.)

- [ ] **Step 4: Route the decode mixin through `_reader`**

In `python/genoray/_svar2_decode.py`, add a `TYPE_CHECKING` method stub so the mixin type-checks in isolation. The block currently reads:

```python
if TYPE_CHECKING:
    import numpy as np
    from seqpro.rag import Ragged
```

Change it to:

```python
if TYPE_CHECKING:
    import numpy as np
    from seqpro.rag import Ragged

    from genoray import _core
```

and in the class body's host-provided declaration block (after `path: Any`), add:

```python
    def _reader(self, contig: str) -> "_core.PyContigReader":  # host-provided
        ...
```

Then in `decode`, replace:

```python
        reader = self._readers[contig]
```
with:
```python
        reader = self._reader(contig)
```

and in `region_counts`, replace:

```python
        flat = self._readers[contig].region_counts(reg)
```
with:
```python
        flat = self._reader(contig).region_counts(reg)
```

- [ ] **Step 5: Route the batch mixin through `_reader`**

In `python/genoray/_svar2_batch.py`, extend the `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from genoray import _core
```

and add to the host-provided declaration block (after `available_samples: list[str]`):

```python
    def _reader(self, contig: str) -> "_core.PyContigReader":  # host-provided
        ...
```

Replace `self._readers[contig]` with `self._reader(contig)` in all four methods:
- `_overlap_batch`: `return self._reader(contig).overlap_batch(reg)`
- `read_ranges`: `return self._reader(contig).read_ranges(reg, self._sample_idxs(samples))`
- `_find_ranges`: `d = self._reader(contig).find_ranges(reg, self._sample_idxs(samples))`
- `_gather_ranges`: `return self._reader(contig).gather_ranges(ranges)`

- [ ] **Step 6: Run the new test to verify it passes**

Run: `pixi run pytest tests/test_svar2_contig_alias.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Run the existing reader suites for regressions**

Run: `pixi run pytest tests/test_svar2_decode.py tests/test_svar2_batch.py tests/test_svar2_ranges.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add python/genoray/_svar2.py python/genoray/_svar2_decode.py \
  python/genoray/_svar2_batch.py tests/test_svar2_contig_alias.py
git commit -m "feat(svar2): normalize contig names in reader query methods"
```

---

## Task 2: Normalize `subset_contigs`

**Files:**
- Modify: `python/genoray/_svar2.py` (`SparseVar2.subset_contigs` ~line 365-384)
- Modify: `tests/test_svar2_concat_split.py:42` (error-message match)
- Test: `tests/test_svar2_contig_alias.py` (append)

**Interfaces:**
- Consumes: `SparseVar2._resolve_contigs` (Task 1).

- [ ] **Step 1: Write the failing test (append to `tests/test_svar2_contig_alias.py`)**

```python
def test_subset_contigs_accepts_unprefixed_contig(tmp_path, svar2_store):
    sv = SparseVar2(svar2_store)  # store contig is "chr1"
    out = tmp_path / "subset.svar2"
    sv.subset_contigs(out, ["1"], overwrite=True)  # unprefixed alias
    assert SparseVar2(out).contigs == ["chr1"]  # canonical store spelling preserved


def test_subset_contigs_unknown_raises(tmp_path, svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="not found in store"):
        sv.subset_contigs(tmp_path / "x.svar2", ["chrZ"], overwrite=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_contig_alias.py::test_subset_contigs_accepts_unprefixed_contig -v`
Expected: FAIL — `ValueError: contigs not in store: ['1']`.

- [ ] **Step 3: Resolve requested contigs in `subset_contigs`**

Replace the body of `subset_contigs` (the `wanted`/`missing`/`kept` block):

```python
        output = Path(output)
        wanted = [contigs] if isinstance(contigs, str) else list(contigs)
        missing = [c for c in wanted if c not in self.contigs]
        if missing:
            raise ValueError(f"contigs not in store: {missing}")
        if output.resolve() == self.path.resolve():
            raise ValueError("cannot write a subset in place (output == source)")
        kept = [c for c in self.contigs if c in set(wanted)]  # preserve source order
```

with:

```python
        output = Path(output)
        wanted = [contigs] if isinstance(contigs, str) else list(contigs)
        resolved = set(self._resolve_contigs(wanted))  # raises ValueError on any miss
        if output.resolve() == self.path.resolve():
            raise ValueError("cannot write a subset in place (output == source)")
        kept = [c for c in self.contigs if c in resolved]  # preserve source order
```

- [ ] **Step 4: Update the existing error-message assertion**

In `tests/test_svar2_concat_split.py:42`, change:

```python
    with pytest.raises(ValueError, match="not in store"):
```
to:
```python
    with pytest.raises(ValueError, match="not found in store"):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_svar2_contig_alias.py tests/test_svar2_concat_split.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2.py tests/test_svar2_contig_alias.py tests/test_svar2_concat_split.py
git commit -m "feat(svar2): normalize contig names in subset_contigs"
```

---

## Task 3: Normalize `annotate_mutations(contigs=)`

**Files:**
- Modify: `python/genoray/_svar2_mutcat.py` (`_MutcatMixin` host-decl block ~line 35-38; `annotate_mutations` scope ~line 69-79)
- Test: `tests/test_svar2_mutcat.py` (append)

**Interfaces:**
- Consumes: `SparseVar2._cnorm` (Task 1).

- [ ] **Step 1: Write the failing test (append to `tests/test_svar2_mutcat.py`)**

Reuse the module-level `_write_parity_ref` / `_write_parity_vcf` helpers already
in `tests/test_svar2_mutcat.py` (both build a `chr1`-named FASTA + VCF). Ensure
`import pytest` and `from pathlib import Path` are present in the file (they are).
Append:

```python
def test_annotate_mutations_contigs_accepts_alias(tmp_path: Path):
    fa = _write_parity_ref(tmp_path)   # chr1-named FASTA
    vcf = _write_parity_vcf(tmp_path)  # chr1 VCF
    out = tmp_path / "v2_alias.svar2"
    SparseVar2.from_vcf(out, vcf, fa, overwrite=True, threads=1)
    sv = SparseVar2(out)               # store contig is "chr1"
    sv.annotate_mutations(fa, contigs=["1"])  # unprefixed alias resolves to chr1
    assert sv._is_annotated()


def test_annotate_mutations_contigs_all_miss_raises(tmp_path: Path):
    fa = _write_parity_ref(tmp_path)
    vcf = _write_parity_vcf(tmp_path)
    out = tmp_path / "v2_miss.svar2"
    SparseVar2.from_vcf(out, vcf, fa, overwrite=True, threads=1)
    sv = SparseVar2(out)
    with pytest.raises(ValueError, match="resolve to a store contig"):
        sv.annotate_mutations(fa, contigs=["chrZ"])
```

(`SparseVar2` is already imported in this file; if not, add `from genoray import SparseVar2`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar2_mutcat.py -k "contigs_accepts_alias or all_miss" -v`
Expected: FAIL — alias case silently annotates nothing (`_is_annotated()` False / no raise), all-miss case does not raise.

- [ ] **Step 3: Declare `_cnorm` on the mutcat mixin**

In `python/genoray/_svar2_mutcat.py`, the host-provided declaration block reads:

```python
    path: Path
    contigs: list[str]
    available_samples: list[str]
    _readers: dict[str, Any]
```

Add `_cnorm` (its type is already imported at top of the module):

```python
    path: Path
    contigs: list[str]
    available_samples: list[str]
    _readers: dict[str, Any]
    _cnorm: ContigNormalizer  # host-provided (SparseVar2._cnorm)
```

- [ ] **Step 4: Resolve `contigs=` in `annotate_mutations`**

Replace the scope block:

```python
        scope = (
            self.contigs
            if contigs is None
            else [c for c in self.contigs if c in set(contigs)]
        )
```

with:

```python
        if contigs is None:
            scope = self.contigs
        else:
            wanted = {n for c in contigs if (n := self._cnorm.norm(c)) is not None}
            scope = [c for c in self.contigs if c in wanted]
            if not scope:
                raise ValueError(
                    f"None of contigs={list(contigs)} resolve to a store contig; "
                    f"available contigs: {self.contigs}"
                )
```

Also (boy-scout, same file, GTF branch ~line 79) reuse the shared normalizer — replace:

```python
            c_norm = ContigNormalizer(self.contigs)
```
with:
```python
            c_norm = self._cnorm
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run pytest tests/test_svar2_mutcat.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genoray/_svar2_mutcat.py tests/test_svar2_mutcat.py
git commit -m "feat(svar2): normalize contigs= in annotate_mutations; raise on all-miss"
```

---

## Task 4: Python↔Rust resolver parity table

**Files:**
- Test: `tests/test_contig_resolver_parity.py` (new)

**Interfaces:**
- Produces: `PARITY_CASES` — the shared `(fasta_contigs, query, expected)` table Task 5's Rust unit test mirrors verbatim.

- [ ] **Step 1: Write the parity test**

Create `tests/test_contig_resolver_parity.py`:

```python
"""Anti-drift guard: the Rust FASTA contig resolver (src/vcf_reader.rs
`resolve_contig_name`) MUST agree with genoray's Python ContigNormalizer on this
shared table. The identical table is asserted in Rust in vcf_reader.rs's tests
module; keep the two in lockstep.
"""

import pytest

from genoray._contigs import ContigNormalizer

# (fasta_contigs, query, expected_or_None)
PARITY_CASES = [
    (["chr1", "chr2", "chrM"], "1", "chr1"),
    (["chr1", "chr2", "chrM"], "chr1", "chr1"),
    (["1", "2", "MT"], "chr1", "1"),
    (["1", "2", "MT"], "1", "1"),
    (["chr1", "chrM"], "MT", "chrM"),   # mito alias
    (["chr1", "chrM"], "chrMT", "chrM"),
    (["1", "MT"], "chrM", "MT"),
    (["chr1", "chr2"], "chrZ", None),   # genuine miss
    (["chr1", "chr2"], "MT", None),     # no mito in reference
]


@pytest.mark.parametrize("contigs,query,expected", PARITY_CASES)
def test_contignormalizer_matches_parity_table(contigs, query, expected):
    assert ContigNormalizer(contigs).norm(query) == expected
```

- [ ] **Step 2: Run to verify it passes (documents the contract)**

Run: `pixi run pytest tests/test_contig_resolver_parity.py -v`
Expected: PASS (9 cases). If any case fails, `PARITY_CASES` misstates `ContigNormalizer`'s behavior — fix the table (it is the source of truth for Task 5).

- [ ] **Step 3: Commit**

```bash
git add tests/test_contig_resolver_parity.py
git commit -m "test(svar2): shared contig-resolver parity table (Python side)"
```

---

## Task 5: Rust FASTA contig resolver

**Files:**
- Modify: `src/vcf_reader.rs` (add `MITO_ALIASES`, `resolve_contig_name`, `resolve_fasta_contig`; wire into `load_contig_seq` + `validate_contigs_in_fasta`; add `#[cfg(test)]` cases)

**Interfaces:**
- Produces: `fn resolve_contig_name(fasta_contigs: &[String], query: &str) -> Option<String>` (pure, testable without I/O); `fn resolve_fasta_contig(fasta: &faidx::Reader, query: &str) -> Option<String>`. Mirrors `PARITY_CASES` from Task 4.

- [ ] **Step 1: Export a non-NFS cargo target dir**

Run: `export CARGO_TARGET_DIR=/tmp/genoray-target-$USER`
(Keep this set for every `cargo`/hook invocation in this task; NFS `target/` bus-errors the linker.)

- [ ] **Step 2: Write the failing Rust unit test**

In `src/vcf_reader.rs`, inside (or add) the `#[cfg(test)] mod tests { ... }` block:

```rust
    // MUST match tests/test_contig_resolver_parity.py::PARITY_CASES verbatim.
    #[test]
    fn resolve_contig_name_parity() {
        let cases: &[(&[&str], &str, Option<&str>)] = &[
            (&["chr1", "chr2", "chrM"], "1", Some("chr1")),
            (&["chr1", "chr2", "chrM"], "chr1", Some("chr1")),
            (&["1", "2", "MT"], "chr1", Some("1")),
            (&["1", "2", "MT"], "1", Some("1")),
            (&["chr1", "chrM"], "MT", Some("chrM")),
            (&["chr1", "chrM"], "chrMT", Some("chrM")),
            (&["1", "MT"], "chrM", Some("MT")),
            (&["chr1", "chr2"], "chrZ", None),
            (&["chr1", "chr2"], "MT", None),
        ];
        for (contigs, query, expected) in cases {
            let owned: Vec<String> = contigs.iter().map(|s| s.to_string()).collect();
            assert_eq!(
                super::resolve_contig_name(&owned, query).as_deref(),
                *expected,
                "query {query:?} against {contigs:?}"
            );
        }
    }
```

- [ ] **Step 3: Run to verify it fails**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-$USER cargo test --no-default-features --features conversion resolve_contig_name_parity'`
Expected: FAIL — `resolve_contig_name` does not exist (compile error).

- [ ] **Step 4: Implement the resolver**

In `src/vcf_reader.rs`, above `load_contig_seq`:

```rust
/// The four mutually-equivalent mitochondrial spellings (mirrors
/// `genoray._contigs._MITO_ALIASES`).
const MITO_ALIASES: [&str; 4] = ["M", "MT", "chrM", "chrMT"];

/// Resolve `query` to the FASTA's own spelling of the same contig, mirroring
/// Python's `ContigNormalizer`: exact match wins, then `chr`-prefix add/strip,
/// then the {M, MT, chrM, chrMT} mito-alias group. Returns `None` if no contig
/// in `fasta_contigs` is equivalent. Pure (no I/O) so it is unit-testable.
fn resolve_contig_name(fasta_contigs: &[String], query: &str) -> Option<String> {
    use std::collections::HashMap;

    // Build the alias -> canonical map in the SAME precedence order as
    // ContigNormalizer.__init__'s dict merge: derived first, exact next
    // (overwrites derived), mito last (overwrites exact for the four aliases).
    let mut m: HashMap<String, String> = HashMap::new();
    for c in fasta_contigs {
        if let Some(stripped) = c.strip_prefix("chr") {
            m.insert(stripped.to_string(), c.clone()); // "chr1" registers "1"
        } else {
            m.insert(format!("chr{c}"), c.clone()); // "1" registers "chr1"
        }
    }
    for c in fasta_contigs {
        m.insert(c.clone(), c.clone()); // exact spelling wins over derived
    }
    if let Some(mito) = fasta_contigs
        .iter()
        .find(|c| MITO_ALIASES.contains(&c.as_str()))
    {
        for a in MITO_ALIASES {
            m.insert(a.to_string(), mito.clone()); // mito group wins over exact
        }
    }
    m.get(query).cloned()
}

/// FASTA-backed wrapper over `resolve_contig_name`.
fn resolve_fasta_contig(fasta: &rust_htslib::faidx::Reader, query: &str) -> Option<String> {
    let names = fasta.seq_names().ok()?;
    resolve_contig_name(&names, query)
}
```

- [ ] **Step 5: Wire the resolver into both fetchers**

In `load_contig_seq`, after opening `fasta` and before `fetch_seq_len(chrom)`, resolve the name:

```rust
    let resolved = match resolve_fasta_contig(&fasta, chrom) {
        Some(name) => name,
        None => {
            return Err(ConversionError::Input(format!(
                "Contig '{chrom}' not found in reference FASTA"
            )));
        }
    };
```

Then use `resolved.as_str()` in place of `chrom` for `fetch_seq_len` and `fetch_seq`:

```rust
    let contig_len_raw = fasta.fetch_seq_len(resolved.as_str());
    // ... (drop the now-unreachable u64::MAX branch's message duplication is fine to keep) ...
    fasta.fetch_seq(resolved.as_str(), 0, contig_len - 1)
```

Keep the `format!("fetching contig '{chrom}' ...")` I/O-error context reporting the caller's `chrom` (not `resolved`) so messages stay caller-facing.

In `validate_contigs_in_fasta`, replace the per-contig check:

```rust
    for chrom in chroms {
        if fasta.fetch_seq_len(chrom.as_str()) == u64::MAX {
            return Err(ConversionError::Input(format!(
                "Contig '{chrom}' not found in reference FASTA"
            )));
        }
    }
```

with:

```rust
    for chrom in chroms {
        if resolve_fasta_contig(&fasta, chrom.as_str()).is_none() {
            return Err(ConversionError::Input(format!(
                "Contig '{chrom}' not found in reference FASTA"
            )));
        }
    }
```

- [ ] **Step 6: Run the Rust test to verify it passes**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-$USER cargo test --no-default-features --features conversion resolve_contig_name_parity'`
Expected: PASS.

- [ ] **Step 7: Run the reference/conversion Rust suite for regressions**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-$USER cargo test --no-default-features --features conversion vcf_reader'`
Expected: PASS (existing `load_contig_seq`/`validate_contigs_in_fasta` tests unaffected — matched-naming paths resolve to the identity).

- [ ] **Step 8: Commit**

```bash
git add src/vcf_reader.rs
git commit -m "fix(svar2): normalize contig names against reference FASTA in conversion"
```

---

## Task 6: Rebuild extension + Python e2e (mismatched FASTA naming)

**Files:**
- Test: `tests/test_svar2_contig_alias.py` (append) — or `tests/test_svar2_from_vcf.py` if the reviewer prefers colocating with conversion tests.

**Interfaces:**
- Consumes: Task 5 (Rust fix).

- [ ] **Step 1: Rebuild the release extension (Task 5's Rust `.so`)**

Run: `pixi run maturin develop --release`
Expected: builds a ~4MB release `.so` (not the ~79MB debug one). `pixi run test` does NOT rebuild it, so this must run before the Python e2e below.

- [ ] **Step 2: Write the failing e2e test**

Append to `tests/test_svar2_contig_alias.py` (self-contained — inlines its own 40 bp reference so it does not depend on `tests/conftest.py` internals):

```python
import subprocess

# 40 bp reference (matches the REF bases used below); kept local to avoid a
# fragile cross-module import of conftest internals.
_REF40 = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _bgzip_index(vcf_path):
    gz = vcf_path.with_suffix(vcf_path.suffix + ".gz")
    subprocess.run(f"bgzip -c {vcf_path} > {gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    return gz


def test_from_vcf_reference_naming_mismatch(tmp_path):
    """Unprefixed VCF contigs ('1') convert against a chr-prefixed FASTA ('chr1')."""
    # chr-prefixed FASTA
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF40}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    # UNPREFIXED VCF (contig "1")
    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\n"  # indel -> exercises left-align vs FASTA
    )
    vcf_gz = _bgzip_index(vcf)

    out = tmp_path / "store.svar2"
    from genoray import SparseVar2

    # Before the fix this raised: "Contig '1' not found in reference FASTA".
    SparseVar2.from_vcf(out, vcf_gz, ref, threads=1, overwrite=True)
    sv = SparseVar2(out)
    assert sv.contigs == ["1"]                 # store keeps the source's spelling
    counts = sv.region_counts("1", [(0, 40)])  # non-empty => indel normalized OK
    assert int(counts.sum()) > 0
```

- [ ] **Step 3: Run to verify it passes (fix in place)**

Run: `pixi run pytest tests/test_svar2_contig_alias.py::test_from_vcf_reference_naming_mismatch -v`
Expected: PASS. (If it errors with `Contig '1' not found in reference FASTA`, the `.so` was not rebuilt — re-run Step 1.)

- [ ] **Step 4: Run the full conversion suite for regressions**

Run: `pixi run pytest tests/test_svar2_from_vcf.py tests/test_svar2_from_svar1.py tests/test_svar2_from_pgen.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_svar2_contig_alias.py
git commit -m "test(svar2): e2e conversion with mismatched reference FASTA naming"
```

---

## Task 7: Update the public-API skill doc

**Files:**
- Modify: `skills/genoray-api/SKILL.md`

**Interfaces:**
- Consumes: final behavior from Tasks 1-6.

- [ ] **Step 1: Locate the relevant sections**

Run: `grep -n "decode\|region_counts\|read_ranges\|subset_contigs\|annotate_mutations\|reference\|ContigNormalizer\|contig" skills/genoray-api/SKILL.md`
Identify where the `SparseVar2` reader methods and conversion `reference=` are documented.

- [ ] **Step 2: Document the reader-side normalization**

Add to the `SparseVar2` reader-methods documentation (near `decode`/`region_counts`/`read_ranges`/`subset_contigs`/`annotate_mutations`) a sentence such as:

> Contig arguments accept alternate naming schemes — `chr`-prefixed vs unprefixed (`chr1` ↔ `1`) and the mitochondrial aliases `{M, MT, chrM, chrMT}` — resolved via `ContigNormalizer` to the store's own spelling. An unresolvable contig raises `ValueError`.

- [ ] **Step 3: Document reference-FASTA normalization**

Near the conversion (`from_vcf`/`from_pgen`/`from_svar1`) `reference=` documentation, add:

> The `reference=` FASTA may use a different contig naming scheme than the variant source; genoray resolves the source's contig names against the FASTA's own naming (`chr`-prefix and mito aliases included).

- [ ] **Step 4: Verify no stale claims + commit**

Confirm no existing SKILL.md text claims contig names must match exactly. Then:

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(svar2): SKILL.md — contig-name normalization for readers and reference"
```

---

## Final verification (after all tasks)

- [ ] **Rebuild + full suite**

Run:
```bash
pixi run maturin develop --release
pixi run test
```
Expected: PASS (full Python suite, including the new alias/e2e tests).

- [ ] **Rust suite**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-$USER cargo test --no-default-features --features conversion'`
Expected: PASS.

- [ ] **Core-build gate (module gating unchanged, but cheap to confirm)**

Run: `pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-$USER cargo check --no-default-features'`
Expected: PASS.

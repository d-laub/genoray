# SVAR write CLI: symbolic + breakend filter flags — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `--skip-symbolic-alts` flag on `genoray write` with two independent, composable flags `--no-symbolic` and `--no-breakend` so users can produce SVARs free of un-injectable ALT classes.

**Architecture:** Two private record-level predicates in `genoray/exprs.py` mirror the existing `is_symbolic` / `is_breakend` polars exprs (anchoring cyvcf2↔polars parity). The CLI `write` command composes whichever filters the flags request into a paired `(cyvcf2 callable, polars expr)` for VCF and a single polars expr for PGEN, then hands them to `SparseVar.from_vcf` / `from_pgen` (which inherit the source filter).

**Tech Stack:** Python, cyclopts (CLI), polars (filter exprs), cyvcf2 (VCF record callable), pgenlib/plink2 (PGEN), vcfixture (test fixtures), pytest, Pixi for the env.

**Spec:** `docs/superpowers/specs/2026-06-05-svar-cli-filter-flags-design.md`

**Run tests with:** `pixi run pytest <path>` (single file/test). Full suite + data regen: `pixi run test`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `genoray/exprs.py` | Add `import re`; add private record-level predicates `_record_is_symbolic`, `_record_is_breakend` co-located with the exprs they mirror | Modify |
| `genoray/_cli/__main__.py` | Replace `skip_symbolic_alts` param with `no_symbolic` + `no_breakend`; compose filters from them | Modify |
| `tests/test_svar_filtering.py` | Rename flag in existing CLI tests; add breakend fixtures + breakend/combined tests | Modify |

---

## Task 1: Record-level parity predicates in `exprs.py`

**Files:**
- Modify: `genoray/exprs.py` (add `import re` near the top with the other imports; add the two helpers immediately after the `is_breakend` definition, ~line 112)
- Test: `tests/test_svar_filtering.py`

These private helpers let the VCF cyvcf2 `filter` callable mirror the `is_symbolic` / `is_breakend` polars exprs without duplicating the `_BND_PATTERN` regex in the CLI.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_filtering.py`:

```python
def test_record_predicates_mirror_exprs():
    from genoray.exprs import _record_is_breakend, _record_is_symbolic

    # symbolic: any ALT starting with "<"
    assert _record_is_symbolic(["<DEL>"]) is True
    assert _record_is_symbolic(["A", "<INS>"]) is True
    assert _record_is_symbolic(["A", "T"]) is False

    # breakend: any ALT in mate-pair or single-breakend notation
    assert _record_is_breakend(["G[chr1:500000["]) is True
    assert _record_is_breakend(["]chr2:321]G"]) is True
    assert _record_is_breakend([".TGCA"]) is True
    assert _record_is_breakend(["TGCA."]) is True
    assert _record_is_breakend(["A", "T"]) is False
    # symbolic alleles are NOT breakends (distinct ALT class)
    assert _record_is_breakend(["<DEL>"]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_filtering.py::test_record_predicates_mirror_exprs -v`
Expected: FAIL with `ImportError: cannot import name '_record_is_breakend'`.

- [ ] **Step 3: Add `import re` to `exprs.py`**

In `genoray/exprs.py`, change the import block (currently just `import polars as pl` at line 16) to:

```python
import re
from collections.abc import Iterable

import polars as pl
```

- [ ] **Step 4: Add the two helpers after `is_breakend`**

In `genoray/exprs.py`, immediately after the `is_breakend` docstring closes (the `"""` ending the block at ~line 112), insert:

```python


def _record_is_symbolic(alts: Iterable[str]) -> bool:
    """Record-level mirror of :data:`is_symbolic` for a cyvcf2 ``Variant.ALT``.

    True if any ALT allele is a symbolic allele (starts with ``<``). Used by the
    CLI to build the cyvcf2 ``filter`` callable that must match the polars
    ``pl_filter`` on the VCF path.
    """
    return any(a.startswith("<") for a in alts)


def _record_is_breakend(alts: Iterable[str]) -> bool:
    """Record-level mirror of :data:`is_breakend` for a cyvcf2 ``Variant.ALT``.

    True if any ALT allele is a breakend (matches :data:`_BND_PATTERN`). Reuses
    the same regex as :data:`is_breakend` so the cyvcf2 ``filter`` callable and
    the polars ``pl_filter`` cannot drift apart.
    """
    return any(re.search(_BND_PATTERN, a) is not None for a in alts)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_filtering.py::test_record_predicates_mirror_exprs -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add genoray/exprs.py tests/test_svar_filtering.py
git commit -m "feat(exprs): add record-level _record_is_symbolic/_record_is_breakend predicates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Replace `skip_symbolic_alts` with `no_symbolic` + `no_breakend` in the CLI

**Files:**
- Modify: `genoray/_cli/__main__.py` (`write`, lines 45-127)
- Test: `tests/test_svar_filtering.py` (existing `test_cli_write_skip_symbolic_*` updated in Task 3)

The current `write` signature has `skip_symbolic_alts: bool = False` (line 53) and an `if skip_symbolic_alts:` branch building the VCF dual-filter (lines 102-111) plus the PGEN `filter=` (line 121). Replace all of it with composition over the two new flags.

- [ ] **Step 1: Update imports and the signature**

In `genoray/_cli/__main__.py`, the parameter list of `write` currently ends with:

```python
    skip_symbolic_alts: bool = False,
) -> None:
```

Replace those lines with:

```python
    no_symbolic: Annotated[bool, Parameter(name="--no-symbolic", negative="")] = False,
    no_breakend: Annotated[bool, Parameter(name="--no-breakend", negative="")] = False,
) -> None:
```

(`Annotated` and `Parameter` are already imported at the top of the file.)

- [ ] **Step 2: Update the docstring**

In the `write` docstring, replace the `skip_symbolic_alts` parameter block (currently lines 75-81, beginning `skip_symbolic_alts` and ending `behavior.`) with:

```
    no_symbolic
        If set, drop records whose ALT contains a symbolic allele
        (``<DEL>``, ``<INS>``, etc.) per VCF 4.x. Applies to both VCF and PGEN
        sources. Recommended for SV-bearing cohorts (e.g. 1kGP SNV_INDEL_SV
        panels) to avoid emitting literal ``<DEL>`` ASCII bytes into downstream
        haplotype buffers.
    no_breakend
        If set, drop records whose ALT contains a breakend (BND) in mate-pair or
        single-breakend notation (``G[chr2:321[``, ``]chr2:321]G``, ``.TGCA``,
        ``TGCA.``). A distinct ALT class from symbolic alleles; combine with
        ``--no-symbolic`` to drop everything haplotype consumers cannot expand.
```

- [ ] **Step 3: Replace the filter construction**

In `genoray/_cli/__main__.py`, after the `if threads is None: threads = -1` block and before `if file_type == "vcf":`, the imports line currently reads:

```python
    from genoray import PGEN, VCF, SparseVar, exprs
    from genoray._utils import variant_file_type
```

Leave that line as-is, then locate the existing VCF branch:

```python
    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        if skip_symbolic_alts:
            # VCF needs both filters: a cyvcf2 callable applied during the
            # genotype scan and a polars expr applied to the index; both must
            # express the same predicate.
            vcf = VCF(
                source,
                dosage_field=dosages,
                filter=lambda rec: not any(a.startswith("<") for a in rec.ALT),
                pl_filter=~exprs.is_symbolic,
            )
        else:
            vcf = VCF(source, dosage_field=dosages)
        SparseVar.from_vcf(
            out, vcf, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=(~exprs.is_symbolic) if skip_symbolic_alts else None,
        )
        SparseVar.from_pgen(
            out, pgen, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")
```

Replace that entire block (from `if file_type == "vcf":` through the final `raise ValueError(f"Unsupported file type: {source}")`) with:

```python
    # Compose the requested ALT-class filters. Each flag contributes a polars
    # expr (used by both VCF index and PGEN) and a record-level predicate (used
    # by the VCF cyvcf2 genotype scan). The two representations of each flag are
    # kept in parity via genoray.exprs.
    pl_terms = []
    record_preds = []
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

        def record_filter(rec, _preds=tuple(record_preds)) -> bool:
            return not any(pred(rec.ALT) for pred in _preds)
    else:
        pl_filter = None
        record_filter = None

    if file_type == "vcf":
        if dosages is not None and Path(dosages).exists():
            raise ValueError(
                "The `dosages` argument appears to be a path to an existing file, but VCF requires a FORMAT field name."
            )

        # VCF requires both filters together (or neither): a cyvcf2 callable
        # applied during the genotype scan and a matching polars expr applied to
        # the index; both must express the same predicate.
        vcf = VCF(
            source,
            dosage_field=dosages,
            filter=record_filter,
            pl_filter=pl_filter,
        )
        SparseVar.from_vcf(
            out, vcf, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
    elif file_type == "pgen":
        pgen = PGEN(
            source,
            dosage_path=dosages,
            filter=pl_filter,
        )
        SparseVar.from_pgen(
            out, pgen, max_mem, overwrite, with_dosages=with_dosages, n_jobs=threads
        )
    else:
        raise ValueError(f"Unsupported file type: {source}")
```

Note: `record_filter`'s keep-predicate is `not any(...)` — a record is kept only if *none* of its ALTs match *any* requested drop-predicate, which is exactly the AND of the negated polars terms.

- [ ] **Step 4: Verify the CLI imports and help still work**

Run: `pixi run genoray write --help`
Expected: help text shows `--no-symbolic` and `--no-breakend` (and NOT `--skip-symbolic-alts`, `--no-no-symbolic`, or `--no-no-breakend`).

- [ ] **Step 5: Commit**

```bash
git add genoray/_cli/__main__.py
git commit -m "feat(cli): replace --skip-symbolic-alts with --no-symbolic and --no-breakend

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Update existing CLI tests for the renamed flag

**Files:**
- Modify: `tests/test_svar_filtering.py` (`test_cli_write_skip_symbolic_vcf` line 178, `test_cli_write_skip_symbolic_pgen` line 189)

- [ ] **Step 1: Rename the flag in both existing CLI tests**

In `tests/test_svar_filtering.py`, change the call in `test_cli_write_skip_symbolic_vcf`:

```python
    cli_write(vcf_path, out, max_mem="1g", overwrite=True, skip_symbolic_alts=True)
```

to:

```python
    cli_write(vcf_path, out, max_mem="1g", overwrite=True, no_symbolic=True)
```

And the same change in `test_cli_write_skip_symbolic_pgen`:

```python
    cli_write(pgen_path, out, max_mem="1g", overwrite=True, no_symbolic=True)
```

- [ ] **Step 2: Run both tests to verify they pass**

Run: `pixi run pytest tests/test_svar_filtering.py -k "cli_write_skip_symbolic" -v`
Expected: PASS (2 passed). `test_cli_write_skip_symbolic_pgen` may be skipped if `plink2` is unavailable.

- [ ] **Step 3: Commit**

```bash
git add tests/test_svar_filtering.py
git commit -m "test(cli): use no_symbolic in renamed-flag CLI tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Breakend fixtures + `--no-breakend` tests

**Files:**
- Modify: `tests/test_svar_filtering.py` (add `Bnd` to the vcfixture import on line 16; add fixtures + tests)

The existing file imports `from vcfixture import Number, Seq, Sym, Type, VcfBuilder, VcfVersion` (line 16). `Bnd(raw: str)` builds a breakend ALT.

- [ ] **Step 1: Add `Bnd` to the vcfixture import**

In `tests/test_svar_filtering.py`, change line 16:

```python
from vcfixture import Number, Seq, Sym, Type, VcfBuilder, VcfVersion
```

to:

```python
from vcfixture import Bnd, Number, Seq, Sym, Type, VcfBuilder, VcfVersion
```

- [ ] **Step 2: Write the failing tests (fixtures + breakend + combined)**

Append to `tests/test_svar_filtering.py`:

```python
def _bnd_vcf(tmp_path: Path) -> Path:
    """chr1: SNV A>T@100, BND G[chr1:500000[@200, <DEL>@300, ins G>GAT@400.

    One of each droppable class plus two keepers, so the three filter modes
    (none / no_breakend / no_symbolic / both) all produce distinct counts.
    """
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
    )
    b.record("chr1", 100, ref="A", alt=[Seq("T")], gt=["0|1", "1|1"])
    b.record("chr1", 200, ref="G", alt=[Bnd("G[chr1:500000[")], gt=["0|1", "0|0"])
    b.record(
        "chr1",
        300,
        ref="A",
        alt=[Sym.deletion()],
        gt=["0|0", "0|1"],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [350]},
    )
    b.record("chr1", 400, ref="G", alt=[Seq("GAT")], gt=["1|1", "0|1"])
    return b.write(tmp_path / "bnd.vcf.gz", bgzip=True, index=True)


def _bnd_pgen(tmp_path: Path) -> Path:
    """Convert the BND VCF to PGEN via plink2 (BND/symbolic carried verbatim)."""
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not available")
    vcf_path = _bnd_vcf(tmp_path)
    prefix = tmp_path / "bnd"
    subprocess.run(
        [
            "plink2",
            "--vcf",
            str(vcf_path),
            "--make-pgen",
            "--out",
            str(prefix),
            "--allow-extra-chr",
        ],
        check=True,
        capture_output=True,
    )
    return prefix.with_suffix(".pgen")


def test_cli_write_no_breakend_vcf(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    vcf_path = _bnd_vcf(tmp_path)
    out = tmp_path / "nobnd.svar"
    cli_write(vcf_path, out, max_mem="1g", overwrite=True, no_breakend=True)
    sv = SparseVar(out)
    # BND@200 dropped; SNV@100, <DEL>@300, ins@400 kept
    assert sv.n_variants == 3
    assert sv.index["POS"].to_list() == [100, 300, 400]


def test_cli_write_no_breakend_pgen(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    pgen_path = _bnd_pgen(tmp_path)
    out = tmp_path / "nobnd_pg.svar"
    cli_write(pgen_path, out, max_mem="1g", overwrite=True, no_breakend=True)
    sv = SparseVar(out)
    assert sv.n_variants == 3
    assert set(sv.index["POS"].to_list()) == {100, 300, 400}


def test_cli_write_no_symbolic_and_no_breakend_vcf(tmp_path):
    from genoray._cli.__main__ import write as cli_write

    vcf_path = _bnd_vcf(tmp_path)
    out = tmp_path / "both.svar"
    cli_write(
        vcf_path, out, max_mem="1g", overwrite=True, no_symbolic=True, no_breakend=True
    )
    sv = SparseVar(out)
    # BND@200 and <DEL>@300 both dropped; only plain SNV@100 + ins@400 remain
    assert sv.n_variants == 2
    assert sv.index["POS"].to_list() == [100, 400]


def test_cli_write_no_flags_keeps_all_vcf(tmp_path):
    """Back-compat: neither flag set -> all records written."""
    from genoray._cli.__main__ import write as cli_write

    vcf_path = _bnd_vcf(tmp_path)
    out = tmp_path / "none.svar"
    cli_write(vcf_path, out, max_mem="1g", overwrite=True)
    sv = SparseVar(out)
    assert sv.n_variants == 4
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `pixi run pytest tests/test_svar_filtering.py -k "no_breakend or no_symbolic_and_no_breakend or no_flags_keeps_all" -v`
Expected: PASS (4 passed; the `_pgen` test may be skipped if `plink2` is unavailable).

- [ ] **Step 4: Run the full filtering test file**

Run: `pixi run pytest tests/test_svar_filtering.py -v`
Expected: PASS (all pass; PGEN tests may skip without `plink2`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_svar_filtering.py
git commit -m "test(cli): add breakend + combined-flag SVAR write tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pixi run test`
Expected: PASS (no failures; PGEN-dependent tests skip only if `plink2` is absent).

- [ ] **Step 2: Lint/format check**

Run: `pixi run -- ruff check genoray tests && pixi run -- ruff format --check genoray tests`
Expected: no errors. If `ruff format --check` reports diffs, run `pixi run -- ruff format genoray tests` and amend the relevant commit.

- [ ] **Step 3: Final flag sanity check**

Run: `pixi run genoray write --help`
Expected: `--no-symbolic` and `--no-breakend` present; no `--skip-symbolic-alts`, `--no-no-symbolic`, or `--no-no-breakend`.

---

## Self-Review Notes (for the author, not a step)

- **Spec coverage:** §1 CLI surface → Task 2; §2 filter composition → Task 2; §3 parity helpers → Task 1; §4 tests → Tasks 3 & 4; §5 docs (write docstring) → Task 2 Step 2, SKILL.md intentionally untouched.
- **Non-goals honored:** no `--compat-gvl`, no `--no-imprecise`, no version/CHANGELOG changes.
- **Parity:** `_record_is_symbolic`/`_record_is_breakend` (Task 1) are the only record-level predicates the CLI (Task 2) references — names match exactly.

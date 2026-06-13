# Fix `annotate_mutations` 1-based POS off-by-one (#59) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 1-based→0-based POS off-by-one in `SparseVar.annotate_mutations` so SBS-96/DBS-78/ID-83 catalogues are computed at the correct reference position, add a defensive guard that turns the `KeyError: '<size>:Del:R:-1'` crash into a graceful UNCLASSIFIED + aggregated "wrong reference" warning, and invalidate stale persisted catalogues.

**Architecture:** `classify_variants` (`genoray/_mutcat.py`) becomes the single boundary between genoray's 1-based index convention and 0-based `reference.fetch`, converting `p = POS - 1`. A module-private `_REF_MISMATCH` sentinel returned by `classify_id83` signals a deletion whose deleted unit is absent in the reference; `classify_variants` maps it to UNCLASSIFIED, counts occurrences, and emits one aggregated loguru warning. `MUTCAT_VERSION` bumps to 2 so previously-persisted `mutcat.npy` is detected as stale on load.

**Tech Stack:** Python, NumPy, Polars, loguru, pysam (test fixtures), pytest.

**Spec:** `docs/superpowers/specs/2026-06-12-annotate-mutations-pos-offby-one-design.md`

**Environment:** all commands run inside Pixi: prefix with `pixi run`.

---

## File Structure

- `genoray/_mutcat.py` — Modify: add `loguru` import; add `_REF_MISMATCH` constant; bump `MUTCAT_VERSION`; guard the two deletion repeat-bucket branches in `classify_id83`; convert POS and aggregate the mismatch warning in `classify_variants`.
- `genoray/_svar.py` — Modify: add a staleness warning in `SparseVar.__init__` when a loaded `mutcat` field's `mutcat_version` is older than `MUTCAT_VERSION`.
- `tests/test_mutcat.py` — Modify: add guard/`_REF_MISMATCH` unit tests for `classify_id83`.
- `tests/test_svar_mutations.py` — Modify: add the issue regression test (deletion no longer crashes) and the SNV-context correctness test.
- `tests/test_mutcat_calibration.py` — Modify: remove the manual `- 1` POS conversion (line ~294).

---

## Task 1: Bump `MUTCAT_VERSION` and add staleness warning on load

This task is independent of the off-by-one fix and is committed first.

**Files:**
- Modify: `genoray/_mutcat.py:145`
- Modify: `genoray/_svar.py` (in `SparseVar.__init__`, after `self.fields = {...}` block at `:573-576`)
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_svar_mutations.py`. This builds an `.svar`, annotates it (writes `mutcat.npy` + `mutcat_version`), then rewrites `metadata.json` with an artificially old `mutcat_version=0` and asserts reopening with `fields=["mutcat"]` logs a staleness warning.

```python
def test_mutcat_staleness_warning(tmp_path, capsys):
    import json
    from loguru import logger
    import genoray
    from genoray import SparseVar
    from genoray._reference import Reference

    # --- build a tiny reference + single-SNV VCF ---
    import pysam
    seq = "ACGTACGTACGTACGT"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))

    vcf = tmp_path / "t.vcf"
    h = pysam.VariantHeader()
    h.add_line("##contig=<ID=chr1,length=16>")
    h.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">')
    h.add_sample("S1")
    with pysam.VariantFile(str(vcf), "w", header=h) as vf:
        r = h.new_record(contig="chr1", start=4, alleles=("A", "C"))  # 1-based POS=5
        r.samples["S1"]["GT"] = (0, 1)
        vf.write(r)
    pysam.tabix_index(str(vcf), preset="vcf", force=True)

    svp = tmp_path / "sv.svar"
    genoray.SparseVar.from_vcf(svp, genoray.VCF(str(vcf) + ".gz"), max_mem="1g", overwrite=True)
    sv = SparseVar(svp)
    sv.annotate_mutations(Reference.from_path(fa), write_back=True)

    # corrupt the persisted version to look stale
    meta_path = svp / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["mutcat_version"] = 0
    meta_path.write_text(json.dumps(meta))

    # capture loguru output
    messages = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        SparseVar(svp, fields=["mutcat"])
    finally:
        logger.remove(sink_id)

    assert any("older version" in m for m in messages), messages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_mutations.py::test_mutcat_staleness_warning -v`
Expected: FAIL — no warning emitted (assertion error on `messages`).

- [ ] **Step 3: Bump `MUTCAT_VERSION`**

In `genoray/_mutcat.py:145`:

```python
MUTCAT_VERSION = 2
```

- [ ] **Step 4: Add the staleness warning in `SparseVar.__init__`**

In `genoray/_svar.py`, immediately after the `self.fields = {...}` dict comprehension (currently `:573-576`), add:

```python
        if (
            "mutcat" in (fields or [])
            and metadata.mutcat_version is not None
            and metadata.mutcat_version < MUTCAT_VERSION
        ):
            logger.warning(
                "mutcat field was computed with an older version "
                f"(v{metadata.mutcat_version} < v{MUTCAT_VERSION}); "
                "recompute via annotate_mutations()."
            )
```

`MUTCAT_VERSION` and `logger` are already imported in `genoray/_svar.py` (`:33` and the loguru import).

- [ ] **Step 5: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_mutations.py::test_mutcat_staleness_warning -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add genoray/_mutcat.py genoray/_svar.py tests/test_svar_mutations.py
git commit -m "fix(svar): bump MUTCAT_VERSION and warn on stale persisted mutcat"
```

---

## Task 2: Defensive guard in `classify_id83` for deletion REF/reference mismatch

Add `_REF_MISMATCH` and guard both deletion repeat-bucket branches so a deletion whose deleted unit is absent at `scan_start` returns the sentinel instead of underflowing into `_repeat_bucket(-1)` → `KeyError`.

**Files:**
- Modify: `genoray/_mutcat.py` (add `loguru` import near top `:8-13`; add `_REF_MISMATCH` constant near `:138`; edit `classify_id83` branches at `:273` and `:283`)
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mutcat.py`. `classify_id83` takes `(pos, ref, alt, fetch)` where `fetch(start, end)` returns reference bytes. A deletion of `"GG"` whose downstream reference is all `"A"` (deleted unit absent at `scan_start`) must return `_REF_MISMATCH`. Mirror the existing `_ref_fn` helper pattern (`tests/test_mutcat.py:103`).

```python
from genoray._mutcat import _REF_MISMATCH  # add to existing import block


def test_id83_2bp_deletion_ref_absent_returns_mismatch():
    # REF="CGG" -> ALT="C": deleted unit "GG". Downstream reference is "AAAA..."
    # so the deleted unit is NOT present at scan_start -> REF/reference mismatch.
    downstream = b"A" * 20

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    assert classify_id83(pos=0, ref=b"CGG", alt=b"C", fetch=fetch) == _REF_MISMATCH


def test_id83_1bp_deletion_ref_absent_returns_mismatch():
    # REF="CG" -> ALT="C": deleted unit "G". Downstream reference is "AAAA..."
    downstream = b"A" * 20

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    assert classify_id83(pos=0, ref=b"CG", alt=b"C", fetch=fetch) == _REF_MISMATCH


def test_id83_2bp_deletion_ref_present_classifies():
    # Deleted unit "GG" IS present once downstream -> valid R:0 channel, not mismatch.
    downstream = b"GG" + b"A" * 18

    def fetch(s: int, e: int) -> bytes:
        return downstream[: e - s]

    code = classify_id83(pos=0, ref=b"CGG", alt=b"C", fetch=fetch)
    assert code != _REF_MISMATCH
    assert code == ID83_INDEX["2:Del:R:0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_mutcat.py -k "ref_absent or ref_present" -v`
Expected: FAIL — `ImportError: cannot import name '_REF_MISMATCH'` (the two mismatch tests would otherwise raise `KeyError: '...:Del:R:-1'`).

- [ ] **Step 3: Add `loguru` import**

In `genoray/_mutcat.py`, with the other imports (after `import polars as pl` at `:10`), add:

```python
from loguru import logger
```

- [ ] **Step 4: Add the `_REF_MISMATCH` sentinel**

In `genoray/_mutcat.py`, immediately after the `SENTINELS` dict (`:134-138`), add:

```python
# Internal boundary signal (NOT a public sentinel): a deletion whose deleted unit
# is absent in the reference at scan_start, i.e. REF disagrees with the reference
# genome. classify_variants maps this to UNCLASSIFIED and aggregates a warning.
_REF_MISMATCH = -99
```

- [ ] **Step 5: Guard both deletion repeat-bucket branches in `classify_id83`**

In `genoray/_mutcat.py`, change the 1 bp branch (`:273-274`):

```python
    if ilen == 1:
        base = indel.decode()
        # fold purine to pyrimidine for the 1bp channel
        if base in ("A", "G"):
            base = chr(_comp(ord(base)))
        if is_del and n_rep == 0:
            return _REF_MISMATCH
        rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
        return ID83_INDEX[f"1:{kind}:{base}:{rep}"]
```

And the ≥2 bp branch (`:283-284`):

```python
    if is_del and n_rep == 0:
        return _REF_MISMATCH
    rep = _repeat_bucket(n_rep - 1 if is_del else n_rep)
    return ID83_INDEX[f"{size}:{kind}:R:{rep}"]
```

(Insertions keep `n_rep` without `-1`, so they never underflow and are untouched.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run pytest tests/test_mutcat.py -k "ref_absent or ref_present" -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "fix(mutcat): guard deletion repeat bucket against REF/reference mismatch"
```

---

## Task 3: Convert POS in `classify_variants` + aggregate mismatch warning

Make `classify_variants` the single source of truth: index POS is 1-based, convert to 0-based for `reference.fetch`. Translate `_REF_MISMATCH` to UNCLASSIFIED, count occurrences, emit one aggregated warning.

**Files:**
- Modify: `genoray/_mutcat.py` (`classify_variants`, `:468-504`)
- Modify: `tests/test_mutcat_calibration.py:294`
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_svar_mutations.py`.

Test A is the issue's exact reproducer — the deletion must no longer crash and must classify as a valid ID-83 channel. Test B confirms the SNV trinucleotide context is read at `POS-1`.

```python
def test_issue59_deletion_no_longer_crashes(tmp_path):
    import pysam
    import genoray
    from genoray import SparseVar
    from genoray._reference import Reference
    from genoray._mutcat import ID83_OFFSET, N_CODES

    seq = "ACGTTGCAACGTTGCAAGGCCTTAGCATCGTACGATCGTTAGCCATGACTGACATGCATGC"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))

    anchor0 = 19
    REF, ALT = seq[anchor0:anchor0 + 6], seq[anchor0]  # "CCTTAG" -> "C", 5bp del

    vcf = tmp_path / "t.vcf"
    h = pysam.VariantHeader()
    h.add_line("##contig=<ID=chr1,length=60>")
    h.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">')
    h.add_sample("S1")
    with pysam.VariantFile(str(vcf), "w", header=h) as vf:
        r = h.new_record(contig="chr1", start=anchor0, alleles=(REF, ALT))
        r.samples["S1"]["GT"] = (0, 1)
        vf.write(r)
    pysam.tabix_index(str(vcf), preset="vcf", force=True)

    svp = tmp_path / "sv.svar"
    genoray.SparseVar.from_vcf(svp, genoray.VCF(str(vcf) + ".gz"), max_mem="1g", overwrite=True)
    sv = SparseVar(svp)
    assert sv.index["POS"].to_list() == [20]  # 1-based preserved

    sv.annotate_mutations(Reference.from_path(fa), write_back=False)  # must NOT raise
    codes = sv.fields["mutcat"].data
    # the deletion entry must be a valid ID-83 code, not UNCLASSIFIED/mismatch
    id83_codes = [c for c in codes if ID83_OFFSET <= c < N_CODES]
    assert len(id83_codes) >= 1


def test_classify_variants_snv_context_uses_pos_minus_one(tmp_path):
    import polars as pl
    from genoray._mutcat import classify_variants, classify_sbs96
    from genoray._reference import Reference
    import pysam

    # Reference where the base at 0-based index differs left vs right of the variant,
    # so a +1 shift would pick a different trinucleotide context.
    seq = "AACGTTGCA"
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\n" + seq + "\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    # Variant at 1-based POS=4 => 0-based index 3, REF = seq[3] = "G".
    p0 = 3
    assert seq[p0] == "G"
    index = pl.DataFrame({"CHROM": ["chr1"], "POS": [p0 + 1], "REF": ["G"], "ALT": [["A"]]})
    codes = classify_variants(index, ref)

    five = seq[p0 - 1].encode()
    three = seq[p0 + 1].encode()
    expected = classify_sbs96(five, b"G", b"A", three)
    assert int(codes[0]) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_mutations.py -k "issue59 or snv_context_uses_pos" -v`
Expected: FAIL — `test_issue59...` raises `KeyError: '5:Del:R:-1'`; `test_classify_variants_snv_context...` asserts wrong code (context shifted +1).

- [ ] **Step 3: Convert POS and aggregate the warning in `classify_variants`**

In `genoray/_mutcat.py`, edit `classify_variants` (`:468-504`). Update the docstring and add the conversion + mismatch aggregation:

```python
def classify_variants(index: pl.DataFrame, reference: Reference) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    ``index`` must have columns CHROM, POS (1-based int, VCF convention), REF
    (str), ALT (List[str]; first ALT used). POS is converted to a 0-based
    reference coordinate internally. Reference context is fetched per contig.
    """
    chrom = index["CHROM"].to_numpy()
    pos = index["POS"].to_numpy().astype(np.int64)
    ref = index["REF"].to_list()
    alt0 = index["ALT"].list.first().to_list()

    out = np.full(index.height, SENTINELS["UNCLASSIFIED"], dtype=np.int16)

    n_mismatch = 0
    mismatch_examples: list[str] = []

    for i in range(index.height):
        r = ref[i]
        a = alt0[i]
        if a is None or r is None:
            continue
        rb, ab = r.encode(), a.encode()
        c = str(chrom[i])
        p = int(pos[i]) - 1  # index POS is 1-based (VCF); reference.fetch is 0-based

        if len(rb) == 1 and len(ab) == 1:  # SNV
            five = reference.fetch(c, p - 1, p).tobytes()
            three = reference.fetch(c, p + 1, p + 2).tobytes()
            out[i] = classify_sbs96(five, rb, ab, three)
        elif len(rb) == 2 and len(ab) == 2:  # native MNV doublet
            out[i] = classify_dbs78(rb, ab)
        elif len(rb) != len(ab):  # indel
            _c = c  # capture per-iteration contig for closure

            def _fetch(s: int, e: int, _c: str = _c) -> bytes:
                return reference.fetch(_c, s, e).tobytes()

            code = classify_id83(p, rb, ab, _fetch)
            if code == _REF_MISMATCH:
                n_mismatch += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append(f"{c}:{int(pos[i])}")
                out[i] = SENTINELS["UNCLASSIFIED"]
            else:
                out[i] = code

    if n_mismatch:
        examples = ", ".join(mismatch_examples)
        logger.warning(
            f"{n_mismatch}/{index.height} deletions have REF disagreeing with the "
            f"reference genome at their position (e.g. {examples}) — wrong reference "
            "build? These were marked UNCLASSIFIED."
        )

    return out
```

- [ ] **Step 4: Update the calibration test**

In `tests/test_mutcat_calibration.py:294`, change the POS column so it no longer pre-converts (the conversion now happens inside `classify_variants`):

```python
            "POS": [v[1] for v in VCF_LINES],  # 1-based; classify_variants converts
```

Also update the comment on the preceding line (`:290`) if it states POS must be 0-based:

```python
    # VCF_LINES has 1-based POS; classify_variants now converts to 0-based internally.
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `pixi run pytest tests/test_svar_mutations.py -k "issue59 or snv_context_uses_pos" -v`
Expected: PASS

- [ ] **Step 6: Run the full mutcat + svar test suites to check for regressions**

Run: `pixi run pytest tests/test_mutcat.py tests/test_svar_mutations.py -v`
Expected: PASS (all). If `tests/test_mutcat_calibration.py` runs locally (requires SigProfilerMatrixGenerator), also run `pixi run pytest tests/test_mutcat_calibration.py -v` and expect PASS; otherwise it is skipped.

- [ ] **Step 7: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat_calibration.py tests/test_svar_mutations.py
git commit -m "fix(svar): convert 1-based POS to 0-based in classify_variants (#59)"
```

---

## Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pixi run pytest -m "not network" -q`
Expected: PASS (no failures, no errors). The previously-crashing `KeyError: '5:Del:R:-1'` path is gone.

- [ ] **Step 2: Confirm no public API / SKILL.md drift**

The fix changes no public name and aligns code with the already-documented convention at `skills/genoray-api/SKILL.md:188` (`svar.index.POS` is 1-based). Confirm no SKILL.md edit is needed:

Run: `grep -n "0-based\|1-based" skills/genoray-api/SKILL.md`
Expected: existing lines already state POS is 1-based; no change required.

- [ ] **Step 3: Final commit (if any stray changes)**

```bash
git status
# if clean, nothing to do
```

---

## Self-Review notes

- **Spec coverage:** Change 1 (POS conversion) → Task 3. Change 2 (guard + aggregated warning) → Tasks 2 & 3. Change 3 (MUTCAT_VERSION bump + staleness warning) → Task 1. All four spec test items covered: regression (Task 3 A), SNV context (Task 3 B), guard+warning (Task 2 + Task 3 aggregation), calibration update (Task 3 Step 4).
- **Type consistency:** `_REF_MISMATCH = -99` defined in Task 2, imported/consumed in Task 2 tests and Task 3. `MUTCAT_VERSION` bumped in Task 1, consumed in `_svar.py` (already imported). `classify_id83(pos, ref, alt, fetch)` signature unchanged.
- **Known limitation:** wrong-reference SNV-context corruption remains silent (per spec, out of scope).

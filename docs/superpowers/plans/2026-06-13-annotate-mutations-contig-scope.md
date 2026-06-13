# Contig-scoped `annotate_mutations` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `SparseVar.annotate_mutations` annotate a caller-selected subset of contigs, marking entries on excluded contigs with a distinct `NOT_ANNOTATED` sentinel and persisting the scope in metadata — so scoped analyses no longer crash on contigs absent from the reference.

**Architecture:** Contig scoping integrates with the vectorized `classify_variants` per-contig loop on the `vectorize-mutation-matrices` branch. `classify_variants` gains a `contigs` allowlist; out-of-scope rows initialize to `NOT_ANNOTATED` and their contigs are never fetched. `annotate_mutations` normalizes the allowlist via `ContigNormalizer`, warns on unmatched entries, gates the SNV adjacency mask to in-scope variants, threads the allowlist down, and records the normalized scope in `SparseVarMetadata.mutcat_contigs`. `MUTCAT_VERSION` bumps to 3.

**Tech Stack:** Python, NumPy, Polars, pydantic (`BaseModel`), loguru, pysam (test fixtures), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-annotate-mutations-contig-scope-design.md`

**Branch:** Implement on a branch off `vectorize-mutation-matrices` (its vectorized `classify_variants` at `genoray/_mutcat.py:760` and `annotate_mutations` at `genoray/_svar.py:1471` are the integration targets). The `.claude/worktrees/vectorize-mutation-matrices/` worktree holds that branch.

**Environment:** all commands run inside Pixi — prefix with `pixi run`.

---

## File Structure

- `genoray/_mutcat.py` — Modify: add `NOT_ANNOTATED` to `SENTINELS` (`:135`); bump `MUTCAT_VERSION` to 3 (`:152`); add `contigs` param + scoping to `classify_variants` (`:760`).
- `genoray/_svar.py` — Modify: add `mutcat_contigs` to `SparseVarMetadata` (`:461`); add `contigs` param + scope logic to `annotate_mutations` (`:1471`).
- `skills/genoray-api/SKILL.md` — Modify: document `contigs=`, `NOT_ANNOTATED`, `mutcat_contigs`.
- `tests/test_svar_mutations.py` — Modify: scoping, normalization, warning, raise, adjacency-suppression, metadata round-trip tests.
- `tests/test_mutcat.py` — Modify: `classify_variants(contigs=...)` unit test.

---

## Task 1: Add `NOT_ANNOTATED` sentinel and bump `MUTCAT_VERSION`

This task is independent and committed first. `NOT_ANNOTATED` is negative, so the existing `code < 0` guard in `_count_kernel` already excludes it from counts — no counting change.

**Files:**
- Modify: `genoray/_mutcat.py:135` (SENTINELS) and `genoray/_mutcat.py:152` (MUTCAT_VERSION)
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py`:

```python
def test_not_annotated_sentinel_and_version():
    from genoray._mutcat import SENTINELS, MUTCAT_VERSION

    # distinct from the existing sentinels and from MISSING (-3)
    assert SENTINELS["NOT_ANNOTATED"] == -4
    assert len(set(SENTINELS.values())) == len(SENTINELS)
    # on-disk semantics changed -> version bumped
    assert MUTCAT_VERSION == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_mutcat.py::test_not_annotated_sentinel_and_version -v`
Expected: FAIL — `KeyError: 'NOT_ANNOTATED'` (and/or `MUTCAT_VERSION == 2`).

- [ ] **Step 3: Implement**

In `genoray/_mutcat.py`, extend `SENTINELS` (`:135`):

```python
SENTINELS: dict[str, int] = {
    "DBS_PARTNER": -1,  # 3' half of an adjacency doublet; never counted
    "UNCLASSIFIED": -2,  # symbolic/complex/MNV>2bp/non-ACGT
    "MISSING": -3,
    "NOT_ANNOTATED": -4,  # entry on a contig excluded from the annotation scope
}
```

Bump the version (`:152`):

```python
MUTCAT_VERSION = 3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_mutcat.py::test_not_annotated_sentinel_and_version -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): add NOT_ANNOTATED sentinel and bump MUTCAT_VERSION to 3"
```

---

## Task 2: Add `contigs` allowlist to `classify_variants`

`contigs` is a list of contig names **already matching the index `CHROM` naming scheme** (the caller — `annotate_mutations` — normalizes them; see Task 4), or `None` for all. Out-of-scope rows initialize to `NOT_ANNOTATED`; in-scope rows initialize to `UNCLASSIFIED` (so an in-scope-but-unclassifiable variant stays `UNCLASSIFIED`). The per-contig loop skips contigs not in the allowlist, so excluded contigs are never fetched. An in-scope contig absent from the reference still raises via `reference.contig_array` — unchanged.

**Files:**
- Modify: `genoray/_mutcat.py:760` (`classify_variants`)
- Test: `tests/test_mutcat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mutcat.py` (mirrors the existing reference/index test setup in this file — a small FASTA + a Polars index with `CHROM/POS/REF/ALT`):

```python
def test_classify_variants_contig_scope(tmp_path):
    import pysam
    import polars as pl
    from genoray._reference import Reference
    from genoray._mutcat import classify_variants, SENTINELS

    # two contigs; chr2 is a valid SNV that WOULD classify if in scope
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n>chr2\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    ref = Reference.from_path(fa)

    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr2"],
            "POS": np.array([2, 2], dtype=np.int32),  # 1-based; 0-based idx 1 -> REF=C
            "REF": ["C", "C"],
            "ALT": [["A"], ["A"]],
        }
    )

    # scope to chr1 only: chr2 must be NOT_ANNOTATED, chr1 must be classified
    out = classify_variants(index, ref, contigs=["chr1"])
    assert out[0] >= 0  # chr1 classified to a real SBS code
    assert out[1] == SENTINELS["NOT_ANNOTATED"]

    # contigs=None -> both classified (unchanged behavior)
    out_all = classify_variants(index, ref, contigs=None)
    assert out_all[0] >= 0 and out_all[1] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_mutcat.py::test_classify_variants_contig_scope -v`
Expected: FAIL — `classify_variants()` got an unexpected keyword argument `contigs`.

- [ ] **Step 3: Implement**

In `genoray/_mutcat.py`, change the `classify_variants` signature and the `out`-initialization / loop guard. Replace the signature and the top of the body (`:760-770`):

```python
def classify_variants(
    index: pl.DataFrame,
    reference: Reference,
    contigs: list[str] | None = None,
) -> np.ndarray:
    """Return an int16 array of intrinsic mutation codes, one per row of ``index``.

    Vectorized: SNV->SBS-96 and native 2bp doublet->DBS-78 via numpy; indels->ID-83
    via a parallel numba kernel. ``index`` must have columns CHROM, POS (1-based),
    REF (str), ALT (List[str]; first used). POS is converted to 0-based internally.

    ``contigs`` restricts annotation to the listed contigs (names must match the
    index ``CHROM`` naming scheme exactly; the caller normalizes them). Rows on
    contigs outside the allowlist are set to ``NOT_ANNOTATED`` and their contigs
    are never fetched. ``None`` (default) annotates all contigs.
    """
    n = index.height
    chrom = index["CHROM"].to_numpy()
    if contigs is None:
        scope: set[str] | None = None
        out = np.full(n, SENTINELS["UNCLASSIFIED"], dtype=np.int16)
    else:
        scope = set(map(str, contigs))
        in_scope = np.array([str(c) in scope for c in chrom], dtype=np.bool_)
        out = np.where(
            in_scope, SENTINELS["UNCLASSIFIED"], SENTINELS["NOT_ANNOTATED"]
        ).astype(np.int16)
    if n == 0:
        return out
```

Then delete the now-duplicated `chrom = index["CHROM"].to_numpy()` line that previously sat after the `if n == 0` block (`:772`), and add a skip guard as the first line inside the `for gi, c in enumerate(uniq):` loop (before `rows = ...`, `:789-790`):

```python
    for gi, c in enumerate(uniq):
        if scope is not None and str(c) not in scope:
            continue
        rows = np.nonzero(inv == gi)[0]
        seq = reference.contig_array(str(c))
```

(Everything from `s_rows = ...` onward, and the mismatch warning block, is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_mutcat.py::test_classify_variants_contig_scope -v`
Expected: PASS

- [ ] **Step 5: Run the full mutcat suite to confirm no regression**

Run: `pixi run pytest tests/test_mutcat.py tests/test_mutcat_calibration.py -q`
Expected: PASS (the `contigs=None` path is byte-for-byte the prior behavior, so the differential/oracle tests still pass).

- [ ] **Step 6: Commit**

```bash
git add genoray/_mutcat.py tests/test_mutcat.py
git commit -m "feat(mutcat): contig allowlist in classify_variants -> NOT_ANNOTATED"
```

---

## Task 3: Add `mutcat_contigs` to `SparseVarMetadata`

Backward-compatible: new field has a default, so all existing construction sites keep working.

**Files:**
- Modify: `genoray/_svar.py:461` (`SparseVarMetadata`)
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar_mutations.py`:

```python
def test_metadata_has_mutcat_contigs_default():
    from genoray._svar import SparseVarMetadata

    m = SparseVarMetadata(samples=["s0"], ploidy=1, contigs=["chr1"])
    assert m.mutcat_contigs is None
    # round-trips through JSON
    m2 = SparseVarMetadata.model_validate_json(
        SparseVarMetadata(
            samples=["s0"], ploidy=1, contigs=["chr1"], mutcat_contigs=["chr1"]
        ).model_dump_json()
    )
    assert m2.mutcat_contigs == ["chr1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_svar_mutations.py::test_metadata_has_mutcat_contigs_default -v`
Expected: FAIL — `mutcat_contigs` is an unexpected/unknown field.

- [ ] **Step 3: Implement**

In `genoray/_svar.py`, extend `SparseVarMetadata` (`:461`):

```python
class SparseVarMetadata(BaseModel):
    version: int | None = None
    samples: list[str]
    ploidy: int
    contigs: list[str]
    fields: dict[str, str] = {}  # field_name -> numpy dtype name (e.g. "float32")
    mutcat_version: int | None = None  # set when annotate_mutations has run
    mutcat_contigs: list[str] | None = None  # normalized contigs annotated; None = all
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_svar_mutations.py::test_metadata_has_mutcat_contigs_default -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat(svar): add mutcat_contigs to SparseVarMetadata"
```

---

## Task 4: Add `contigs` to `annotate_mutations` (scope, warn, gate, persist)

Normalize the allowlist via `self._c_norm`, warn on entries matching no index contig, compute the per-variant in-scope mask, gate `is_snv` to in-scope (so out-of-scope adjacent SNVs are not DBS-collapsed), thread the normalized allowlist to `classify_variants`, and persist `mutcat_contigs`.

**Files:**
- Modify: `genoray/_svar.py:1471` (`annotate_mutations`)
- Test: `tests/test_svar_mutations.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_svar_mutations.py`. These reuse the existing `_build_tiny_svar` helper (2 samples, ploidy 1, 3 SNVs on `chr1` at 1-based POS 2,3,9; sample 0 carries the adjacent pair → DBS).

```python
def _build_two_contig_svar(path):
    """chr1 has the 3-SNV tiny layout; chr2 carries one adjacent SNV pair.

    Variants (1-based POS): chr1@2, chr1@3, chr1@9, chr2@2, chr2@3.
    chr2@2 and chr2@3 are adjacent SNVs on sample 0's single haplotype.
    """
    from genoray._svar import SparseVarMetadata, _write_genos
    from seqpro.rag import Ragged

    path.mkdir(parents=True)
    index = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "POS": np.array([2, 3, 9, 2, 3], dtype=np.int32),
            "REF": ["C", "G", "A", "C", "G"],
            "ALT": [["A"], ["T"], ["C"], ["A"], ["T"]],
            "ILEN": pl.Series([[0]] * 5, dtype=pl.List(pl.Int32)),
        }
    )
    index.write_ipc(path / "index.arrow")
    # sample 0 carries chr1@2, chr1@3 (var 0,1) and chr2@2, chr2@3 (var 3,4);
    # sample 1 carries chr1@9 (var 2). ploidy 1 -> 2 tracks.
    data = np.array([0, 1, 3, 4, 2], dtype=np.int32)
    offsets = np.array([0, 4, 5], dtype=np.int64)  # n_samples*ploidy + 1 = 3
    genos = Ragged.from_offsets(data, (2, 1, None), offsets)
    _write_genos(path, genos)
    with open(path / "metadata.json", "w") as f:
        f.write(
            SparseVarMetadata(
                version=1, samples=["s0", "s1"], ploidy=1, contigs=["chr1", "chr2"]
            ).model_dump_json()
        )


def _two_contig_ref(tmp_path):
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nACGTACGTAC\n>chr2\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    return Reference.from_path(fa)


def test_annotate_scope_marks_excluded_not_annotated(tmp_path):
    from genoray._mutcat import SENTINELS

    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(_two_contig_ref(tmp_path), contigs=["chr1"], write_back=True)

    mut = svar.fields["mutcat"]
    # entries on chr2 (sample 0's last two entries: vars 3,4) must be NOT_ANNOTATED
    flat = mut.data
    # data order matches genos: [chr1@2, chr1@3, chr2@2, chr2@3, chr1@9]
    assert flat[2] == SENTINELS["NOT_ANNOTATED"]
    assert flat[3] == SENTINELS["NOT_ANNOTATED"]
    # chr1 entries are NOT NOT_ANNOTATED (classified or DBS)
    assert flat[0] != SENTINELS["NOT_ANNOTATED"]


def test_annotate_scope_excludes_from_matrix(tmp_path):
    """A scoped run's SBS96 matrix equals an all-contig run on a chr1-only svar."""
    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(_two_contig_ref(tmp_path), contigs=["chr1"], write_back=False)
    sbs = svar.mutation_matrix("SBS96", count="allele")
    # chr2's SNVs contribute nothing; totals come only from chr1 isolated SNV(s)
    assert sbs.height == 96
    # chr1@9 (sample 1) is an isolated SNV -> exactly one SBS event total
    total = sum(sbs[c].sum() for c in ["s0", "s1"])
    assert total == 1  # chr1@2,@3 collapse to DBS (not SBS); chr2 excluded


def test_annotate_scope_normalizes_contig_names(tmp_path):
    """Allowlist given without 'chr' prefix still matches a chr-prefixed index."""
    from genoray._mutcat import SENTINELS

    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    # index uses 'chr1'/'chr2'; pass '1' -> should match chr1
    svar.annotate_mutations(_two_contig_ref(tmp_path), contigs=["1"], write_back=False)
    flat = svar.fields["mutcat"].data
    assert flat[2] == SENTINELS["NOT_ANNOTATED"]  # chr2 excluded
    assert flat[0] != SENTINELS["NOT_ANNOTATED"]  # chr1 annotated


def test_annotate_scope_warns_on_unmatched_contig(tmp_path, capsys):
    # loguru writes to stderr by default; capsys captures it (caplog does not,
    # since loguru bypasses stdlib logging) — matches the repo's #59 convention.
    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(
        _two_contig_ref(tmp_path), contigs=["chr1", "chrZ"], write_back=False
    )
    assert "chrZ" in capsys.readouterr().err


def test_annotate_scope_adjacent_oos_snvs_not_dbs(tmp_path):
    """chr2's adjacent SNV pair, when out of scope, is NOT collapsed to DBS."""
    from genoray._mutcat import SENTINELS

    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(_two_contig_ref(tmp_path), contigs=["chr1"], write_back=False)
    flat = svar.fields["mutcat"].data
    # both chr2 entries are NOT_ANNOTATED; neither is a DBS code or DBS_PARTNER
    assert flat[2] == SENTINELS["NOT_ANNOTATED"]
    assert flat[3] == SENTINELS["NOT_ANNOTATED"]
    assert flat[3] != SENTINELS["DBS_PARTNER"]


def test_annotate_scope_persists_mutcat_contigs(tmp_path):
    from genoray._svar import SparseVarMetadata

    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(_two_contig_ref(tmp_path), contigs=["chr1"], write_back=True)
    with open(d / "metadata.json", "rb") as f:
        meta = SparseVarMetadata.model_validate_json(f.read())
    assert meta.mutcat_contigs == ["chr1"]
    assert meta.mutcat_version == 3


def test_annotate_no_scope_persists_none(tmp_path):
    from genoray._svar import SparseVarMetadata

    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    svar.annotate_mutations(_two_contig_ref(tmp_path), write_back=True)
    with open(d / "metadata.json", "rb") as f:
        meta = SparseVarMetadata.model_validate_json(f.read())
    assert meta.mutcat_contigs is None


def test_annotate_inscope_contig_absent_from_reference_raises(tmp_path):
    """Listing a contig present in the index but absent from the reference raises."""
    d = tmp_path / "two.svar"
    _build_two_contig_svar(d)
    svar = SparseVar(d)
    # reference has only chr1; chr2 is in scope but unfetchable
    fa = tmp_path / "ref1.fa"
    fa.write_text(">chr1\nACGTACGTAC\n")
    pysam.faidx(str(fa))
    with pytest.raises(Exception):
        svar.annotate_mutations(Reference.from_path(fa), contigs=["chr1", "chr2"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run pytest tests/test_svar_mutations.py -k "scope or persists" -v`
Expected: FAIL — `annotate_mutations()` got an unexpected keyword argument `contigs`.

- [ ] **Step 3: Implement**

In `genoray/_svar.py`, replace the `annotate_mutations` signature and body up through the `is_snv`/`classify_variants` section (`:1471-1534`). Add the `contigs` parameter, the scope resolution, the `classify_variants` call with `contigs=`, and the `is_snv` gate:

```python
    def annotate_mutations(
        self,
        reference: "Reference | str | Path",
        *,
        contigs: list[str] | None = None,
        write_back: bool = True,
    ) -> None:
        """Classify every variant into SBS-96 / DBS-78 / ID-83 channels and store
        a per-genotype-entry ``mutcat`` field (int16, enum-encoded).

        Adjacent SNVs carried on the same haplotype are combined into DBS; the
        5' entry receives the DBS code and the 3' entry a ``DBS_PARTNER`` sentinel.

        Parameters
        ----------
        reference
            Reference genome.  A :class:`~genoray._reference.Reference` instance,
            or a path to a FASTA file (with a ``.fai`` index alongside it).
        contigs
            If given, only variants on these contigs are classified; entries on
            all other contigs are marked ``NOT_ANNOTATED`` and their contigs are
            never fetched from the reference.  Names are matched via the
            :class:`~genoray._utils.ContigNormalizer` (so ``chr1``/``1`` both
            work).  Requested contigs absent from the ``.svar`` index are skipped
            with a warning.  A listed contig present in the index but absent from
            the reference still raises (use the allowlist to exclude it instead).
            ``None`` (default) classifies all contigs.
        write_back
            If ``True`` (default), persist ``mutcat.npy`` and update
            ``metadata.json`` on disk so that subsequent ``SparseVar(...)``
            opens will see the field.  If ``False``, the ``mutcat`` field lives
            only in memory (``self.fields["mutcat"]``) and is NOT written to
            disk — reopening the file will not find it.  Note: the
            ``metadata.json`` update is not safe against concurrent writers;
            single-writer access is expected (consistent with
            ``annotate_with_gtf``).
        """
        if not isinstance(reference, Reference):
            reference = Reference.from_path(reference)

        # 0. resolve contig scope
        index_chroms = self.index["CHROM"].to_list()
        if contigs is None:
            scoped_contigs: list[str] | None = None
            in_scope = np.ones(self.index.height, dtype=np.bool_)
        else:
            normalized = self._c_norm.norm(list(contigs))
            unmatched = [c for c, nm in zip(contigs, normalized) if nm is None]
            if unmatched:
                logger.warning(
                    f"annotate_mutations: {len(unmatched)} requested contig(s) not "
                    f"found in the .svar index; they will be skipped: {unmatched}"
                )
            scope_set = {nm for nm in normalized if nm is not None}
            scoped_contigs = sorted(scope_set)
            in_scope = np.array([c in scope_set for c in index_chroms], dtype=np.bool_)

        # 1. intrinsic per-variant codes (scoped)
        var_code = classify_variants(self.index, reference, contigs=scoped_contigs)

        # 2. per-variant arrays needed by the adjacency kernel
        pos = self.index["POS"].to_numpy().astype(np.int64)
        ref0 = self.index["REF"].to_list()
        alt0 = self.index["ALT"].list.first().to_list()
        is_snv = np.array(
            [
                r is not None and a is not None and len(r) == 1 and len(a) == 1
                for r, a in zip(ref0, alt0)
            ],
            dtype=np.bool_,
        )
        # out-of-scope variants must not participate in DBS adjacency, so their
        # NOT_ANNOTATED code broadcasts unchanged to every entry.
        is_snv &= in_scope
```

The remainder of the method — the `contig_map`/`contig_codes`/`ref_b`/`alt_b` arrays, `build_entry_codes`, the in-memory field registration, and the `write_back` block — stays as-is, except add the `mutcat_contigs` line in the `write_back` metadata update (after `meta.mutcat_version = MUTCAT_VERSION`):

```python
            meta.fields["mutcat"] = "int16"
            meta.mutcat_version = MUTCAT_VERSION
            meta.mutcat_contigs = scoped_contigs
            with open(self.path / "metadata.json", "w") as f:
                f.write(meta.model_dump_json())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_svar_mutations.py -k "scope or persists" -v`
Expected: PASS

- [ ] **Step 5: Run the full svar-mutations + mutcat suites**

Run: `pixi run pytest tests/test_svar_mutations.py tests/test_mutcat.py tests/test_mutcat_calibration.py -q`
Expected: PASS (existing tests unaffected; `contigs=None` is the prior path).

- [ ] **Step 6: Commit**

```bash
git add genoray/_svar.py tests/test_svar_mutations.py
git commit -m "feat(svar): contig scope + NOT_ANNOTATED + mutcat_contigs in annotate_mutations (#62)"
```

---

## Task 5: Update the public API skill doc

Per `CLAUDE.md`, the same PR must update `skills/genoray-api/SKILL.md` for the new public surface: the `contigs=` kwarg, the `NOT_ANNOTATED` sentinel, and `mutcat_contigs`.

**Files:**
- Modify: `skills/genoray-api/SKILL.md` (the `annotate_mutations` section, around `:304-338`)

- [ ] **Step 1: Update the signature and parameter docs**

Change the signature line to:

```
Signature: `annotate_mutations(reference, *, contigs=None, write_back=True) -> None`
```

Add a bullet describing `contigs` (place it before the `write_back=True` bullet):

```markdown
- `contigs=None` — if given (a list of contig names), only variants on those
  contigs are classified; entries on all other contigs are marked
  `NOT_ANNOTATED` (sentinel `-4`) and their contigs are never fetched from the
  reference. Names match via the `ContigNormalizer` (`chr1`/`1` both work).
  Requested contigs absent from the `.svar` index are skipped with a warning; a
  listed contig present in the index but absent from the reference raises (omit
  it from the list to exclude it cleanly). `None` (default) classifies all
  contigs. When `write_back=True`, the normalized scope is recorded in
  `metadata.json` as `mutcat_contigs` (`None` = all).
```

- [ ] **Step 2: Document the new sentinel in the classification table**

In the "What it classifies" table, add a row:

```markdown
| Variant on a contig outside `contigs=` | `NOT_ANNOTATED` (excluded from all matrices) |
```

- [ ] **Step 3: Verify there are no other stale references**

Run: `grep -n "annotate_mutations" skills/genoray-api/SKILL.md`
Expected: every shown signature/usage reflects the `contigs=` kwarg (the examples that omit it still work, since it defaults to `None`).

- [ ] **Step 4: Commit**

```bash
git add skills/genoray-api/SKILL.md
git commit -m "docs(skill): document contigs= scope and NOT_ANNOTATED in annotate_mutations"
```

---

## Task 6: Full-suite regression check

- [ ] **Step 1: Run the complete test suite**

Run: `pixi run pytest -m "not network" -q`
Expected: PASS — no regressions across the repo.

- [ ] **Step 2: Lint/format**

Run: `pixi run ruff check genoray tests && pixi run ruff format --check genoray tests`
Expected: clean (run `ruff format genoray tests` first if needed, then re-stage/commit).

---

## Self-Review notes

- **Spec coverage:** §1 sentinel → Task 1; §2 `classify_variants(contigs=)` → Task 2; §3 `annotate_mutations` scope/gate/persist → Task 4; §4 metadata + version → Tasks 1 & 3; §6 SKILL.md → Task 5; testing §7 → Tasks 2 & 4. §5 (#61 relationship) is a no-code decision (no auto-skip mode) — honored by `contigs=None` keeping the raise-on-absent path (test in Task 4).
- **Type consistency:** `contigs: list[str] | None` is identical across `classify_variants` and `annotate_mutations`; `scoped_contigs` (normalized, index-scheme names or `None`) is what both `classify_variants` receives and `mutcat_contigs` stores; the `in_scope` bool mask gates `is_snv`. `NOT_ANNOTATED = -4` used consistently.
- **Naming authority:** `annotate_mutations` normalizes via `self._c_norm` and passes already-index-scheme names to `classify_variants`, which does pure set membership (no second normalization, no double warning).

---
name: genoray-api
description: Use when writing or modifying Python code that imports `genoray` to read genotypes/dosages from VCF, PGEN, or SparseVar (`.svar`) files. Covers the public API surface, mode constants, range queries, chunking, filtering, and the SparseVar workflow. Skip for unrelated bioinformatics work.
---

# genoray public API

`genoray` is a NumPy-first range-query layer over VCF/BCF (cyvcf2), PGEN
(pgenlib), and a sparse memmap format (`SparseVar` / `.svar`).

## Public surface

`import genoray` exposes exactly:

- `genoray.PGEN` — PLINK 2 PGEN reader
- `genoray.Reference` — indexed-FASTA reference genome reader
- `genoray.VCF` — VCF/BCF reader
- `genoray.Reader` — type alias `VCF | PGEN | SparseVar`
- `genoray.SparseVar` — sparse `.svar` reader/writer
- `genoray.exprs` — polars filter expressions for `.gvi` indexes
- `genoray.cosmic_signatures` — fetch/cache COSMIC reference signatures
- `genoray.fit_signatures` — sparse forward-selection signature refit

Nothing else is public. Anything starting with `_` (e.g. `genoray._vcf`) is
internal — do not import it from user code.

## Where to look for details

Prefer reading these over guessing:

- `docs/source/index.md` — narrative tour with full examples (VCF, PGEN, filtering, chunking)
- `docs/source/svar.md` — SparseVar usage
- `genoray/__init__.py` — confirms the public surface
- `genoray/_vcf.py` — `VCF` class: constructor, `read`, `chunk`, mode constants near the top of the class
- `genoray/_pgen.py` — `PGEN` class: constructor, `read`, `chunk`, `read_ranges`, `chunk_ranges`, mode constants near the top of the class
- `genoray/_svar.py` — `SparseVar`: `__init__`, `from_vcf`, `from_pgen`, `read_ranges`, `with_fields`, `annotate_mutations`, `mutation_matrix`, `assign_signatures`
- `genoray/_signatures.py` — `cosmic_signatures`, `fit_signatures`
- `genoray/_reference.py` — `Reference`: `from_path`, `fetch`, `contig_array`
- `genoray/exprs.py` — the *complete* set of pre-built filter expressions (currently 7: `is_snp`, `is_indel`, `is_biallelic`, `is_symbolic`, `is_breakend`, `is_imprecise`, `ILEN`)

When a signature, kwarg, or shape is unclear, **read the docstring in the
source** rather than reasoning from first principles.

## Cross-cutting conventions

- Ranges are 0-based, half-open `[start, end)`.
- `max_mem` accepts strings like `"4g"`, `"512m"`, `"2GB"`.
- Contig names auto-normalize: `"chr1"` and `"1"` both work regardless of file convention (`ContigNormalizer`).
- Missing genotype = `-1` (int). Missing dosage = `np.nan` (float32).
- Ploidy is 2 by default; `SparseVar.from_vcf`/`from_pgen` (and `genoray write`) accept `haploid=True` / `--haploid`, which OR-collapses haplotypes into a single haploid call per sample and records `ploidy=1` in metadata (intended for unphased somatic data).
- All return arrays are NumPy; `mode` selects which arrays you get back.

## Mode constants — gotcha

Modes are **class attributes**, not top-level names:

```python
genoray.VCF.Genos8           # not genoray.Genos8
genoray.PGEN.GenosPhasingDosages
```

To discover the available modes for a class, read the class body in
`_vcf.py` / `_pgen.py` (search for `Genos` near the top).

When a mode bundles multiple arrays, the return tuple follows the order in
the constant name. `PGEN.GenosPhasingDosages` returns `(genos, phasing,
dosages)`; `VCF.Genos8Dosages` returns `(genos, dosages)`.

## VCF — quick reference

```python
vcf = genoray.VCF(
    "file.vcf.gz",
    phasing=True,             # constructor-time, not per-read
    dosage_field="DS",        # required to read dosages; FORMAT field with Number=A
    filter=lambda v: ...,     # cyvcf2.Variant -> bool
    pl_filter=~genoray.exprs.is_symbolic,  # drop <DEL>/<INS>/...; pair with a matching `filter` callable
)

# Single range
arr = vcf.read("chr1", start=0, end=1_000_000, mode=genoray.VCF.Genos8)

# Chunked
for chunk in vcf.chunk("chr1", start=0, end=1_000_000,
                       max_mem="2g", mode=genoray.VCF.Genos8Dosages):
    ...
```

- Shape with `phasing=False`: `(samples, ploidy=2, variants)`.
- Shape with `phasing=True`: `(samples, ploidy+1=3, variants)` — the 3rd row along the ploidy axis is `0` (unphased) / `1` (phased), matching cyvcf2.
- Dosage arrays drop the ploidy axis: `(samples, variants)`, dtype `float32`.
- VCF intentionally has **no `read_ranges`** — benchmarking showed no throughput benefit.

## PGEN — quick reference

```python
pgen = genoray.PGEN(
    "hardcalls.pgen",                # hardcalls live in the main path
    dosage_path="dosages.pgen",      # optional; defaults to the main path
    filter=genoray.exprs.is_snp & genoray.exprs.is_biallelic,
)
```

Important: when you have a dosage-only PGEN and a separate hardcalls PGEN,
**hardcalls go in the main path** and dosages go in `dosage_path`. If you
only pass one path, both hardcalls and dosages come from it (with the
hardcalls inferred from dosage threshold — see PLINK 2 docs).

A `.gvi` index file is created next to the PGEN on first construction.
Don't delete it.

```python
# Single range
genos = pgen.read("chr2", start=0, end=1000)

# Multiple ranges in one call (PGEN-only optimization)
data, offsets = pgen.read_ranges(
    "chr2",
    starts=[0, 1000, 2000],
    ends=[1000, 2000, 3000],
    mode=genoray.PGEN.GenosPhasingDosages,
)
# `data` matches the mode (tuple when mode bundles multiple arrays)
# `offsets` shape: (n_ranges + 1,). Slice range i with: arr[..., offsets[i]:offsets[i+1]]

# Chunked variants of both
for chunk in pgen.chunk("chr2", 0, 1000, max_mem="4g"): ...
for range_iter in pgen.chunk_ranges("chr2", starts, ends, max_mem="4g"):
    for chunk in range_iter: ...
```

Genotype dtype: `int32`. Dosage dtype: `float32`. Phasing is a separate
`bool` array of shape `(samples, variants)` — *not* an extra row in the
genotype array (unlike VCF with `phasing=True`).

## SparseVar (`.svar`) — quick reference

Build:

```python
# From a configured VCF reader
vcf = genoray.VCF("file.vcf.gz", dosage_field="DS")
genoray.SparseVar.from_vcf("out.svar", vcf, max_mem="4g",
                           with_dosages=True, overwrite=True)

# Or from a PGEN
genoray.SparseVar.from_pgen("out.svar", "file.pgen", max_mem="4g")

# Unphased somatic data: collapse to a single haploid call per sample (ploidy=1)
genoray.SparseVar.from_vcf("out.svar", vcf, max_mem="4g", haploid=True)
```

`SparseVar.from_vcf` / `from_pgen` inherit and apply the source's filter — filter the VCF/PGEN to filter the SVAR.

`SparseVar.from_vcf` / `from_pgen` accept `regions=`, `samples=`,
`merge_overlapping=`, `regions_overlap=` to subset by region and/or sample
during conversion (same semantics as `SparseVar.write_view`); a sample subset
drops MAC=0 variants from the output.

Read:

```python
# Plain ragged: data is just variant indices
svar = genoray.SparseVar("out.svar")
ragged = svar.read_ranges("chr1", starts=[0, 50_000], ends=[10_000, 60_000],
                          samples=["S1", "S2"])
# shape: (ranges, samples, ploidy, ~variants) — last axis is ragged

# With extra fields attached
svar = genoray.SparseVar("out.svar", fields={"dosages": np.float32})
# or, on an existing instance:
svar_with = svar.with_fields({"dosages": np.float32})
result = svar_with.read_ranges("chr1", [0], [10_000])
result.genos     # Ragged of variant indices (uint32)
result.dosages   # Ragged of dosages (float32)
```

`with_fields(False)` drops all extras and returns a plain
`Ragged[V_IDX_TYPE]` again from subsequent reads.

Each leaf value in the ragged result is a **variant index** — a row number
into `svar.index`, a polars `DataFrame` with at least `CHROM, POS, REF,
ALT (list[str]), ILEN`. To map indices back to chrom/pos/ref/alt, row-index
that DataFrame.

```python
v_idxs = ragged[0, 0, 0].to_numpy()
rows = svar.index[v_idxs.tolist()].select("CHROM", "POS", "REF", "ALT")
```

`svar.index.POS` is **1-based** (VCF convention), while query coordinates
are **0-based half-open**. Don't conflate them.

## Filtering

VCF: pass a `Callable[[cyvcf2.Variant], bool]` to `filter=`. For index-based
predicates (e.g. `is_symbolic`), also pass the matching polars `pl.Expr` to
`pl_filter=` — VCF requires **both** when filtering via the `.gvi` index.

To change a VCF's filter after construction, assign a `(filter, pl_filter)`
tuple to the `vcf.filter` setter (or `None` to clear both); the same
both-or-neither invariant is enforced, and the in-memory index is invalidated.
The getter returns the `(filter, pl_filter)` tuple, mirroring the setter
(`(None, None)` when unset), so `vcf.filter = vcf.filter` round-trips.

```python
vcf.filter = (lambda rec: ..., ~genoray.exprs.is_symbolic)  # set both
vcf.filter = None                                           # clear both
fn, expr = vcf.filter                                       # get both
```

PGEN: pass a polars `pl.Expr` returning a boolean mask, operating on the
`.gvi` index columns. Built-in expressions in `genoray.exprs` (the
*complete* list):

- `is_snp` (True if **all** ALT alleles have ILEN == 0; rows with any `null` ILEN → False)
- `is_indel` (True if **all** ALT alleles have ILEN != 0; rows with any `null` ILEN → False)
- `is_biallelic`
- `is_symbolic` (True if any ALT is a VCF 4.x symbolic allele, i.e. starts with `<`)
- `is_breakend` (True if any ALT is a VCF 4.x breakend in mate-pair / single-breakend notation, e.g. `G[chr2:321[`, `]chr2:321]G`, `.TGCA`, `TGCA.`. A *distinct* ALT class from symbolic alleles — `is_symbolic` does **not** flag breakends)
- `is_imprecise` (True if any ALT's ILEN is `null` — an un-sizable symbolic allele **or** a breakend)
- `ILEN` (a `List[Int32]` expression — one value per ALT allele, not a boolean)

**`ILEN` semantics for symbolic SVs.** For precise `<DEL>`/`<INS>`/`<DUP>`,
ILEN is computed at index-build time from INFO fields: `-|SVLEN|` for `<DEL>`,
`+|SVLEN|` for `<INS>`/`<DUP>` (falls back to `|END - POS|` when SVLEN is absent).
For VCF, INFO fields are read from header-declared columns (via oxbow); for PGEN,
they are parsed from the PVAR INFO string. Non-symbolic ALTs use the literal
`len(ALT) - len(REF)`.

**Un-sizable symbolic alleles carry `null` ILEN.** An allele is un-sizable when:
the `IMPRECISE` INFO flag is set, `SVLEN`/`END` are both missing, the symbolic
type is unsupported (`<BND>`, `<CNV>`, `<INV>`, `<*>`/`<NON_REF>`), or the ALT is
a breakend in mate-pair / single-breakend notation (e.g. `G[chr2:321[`). At NumPy
materialization, `null` ILEN is coerced to 0 (treated as a point variant).

**Filtering guidance** (no new constructor kwarg — use the existing `filter`/`pl_filter` API):

- `~genoray.exprs.is_symbolic` — drops *all* symbolic alleles (precise or not).
  Required for haplotype consumers (e.g. `genvarloader`) that cannot expand any
  symbolic ALT into literal sequence:

  ```python
  # PGEN
  pgen = genoray.PGEN("file.pgen", filter=~genoray.exprs.is_symbolic)
  # VCF (both required)
  vcf = genoray.VCF(
      "file.vcf.gz",
      filter=lambda rec: not any(a.startswith("<") for a in rec.ALT),
      pl_filter=~genoray.exprs.is_symbolic,
  )
  ```

- `~genoray.exprs.is_imprecise` — keeps precise symbolic SVs (correctly
  sized/spanned) and drops only the un-sizable ones (including breakends, which
  are always un-sizable). Suitable for range/overlap queries where precise SVs
  are queryable:

  ```python
  pgen = genoray.PGEN("file.pgen", filter=~genoray.exprs.is_imprecise)
  ```

- For haplotype consumers, drop *all* un-expandable ALTs (symbolic **and**
  breakends) — breakends are not caught by `~is_symbolic`:

  ```python
  hap_safe = ~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend
  pgen = genoray.PGEN("file.pgen", filter=hap_safe)
  ```

For anything else, write `pl.col(...)` against the `.gvi` schema — read
`genoray/exprs.py` for the available columns. Combining two `exprs`
expressions with `&` / `|` works without importing polars; you only need
`import polars as pl` to build custom predicates.

## Reference — quick reference

`genoray.Reference` is a pysam-backed indexed-FASTA reader used to supply
flanking context for mutation-catalogue classification.

```python
ref = genoray.Reference.from_path("hg38.fa")          # auto-creates .fai if absent
ref = genoray.Reference.from_path("hg38.fa", contigs=["chr1", "chr2"])

seq: np.ndarray = ref.fetch("chr1", start=1_000_000, end=1_000_010)
# returns uint8 NDArray, 0-based half-open [start, end)
# bytes(seq) gives the ASCII sequence
```

Key properties:

- `from_path(fasta, contigs=None)` — `fasta` is a `str | Path`; auto-calls `pysam.faidx` if the `.fai` index is missing. `contigs` filters which contigs the caller cares about (defaults to all in the FASTA).
- `fetch(contig, start, end)` — 0-based half-open `[start, end)`. Positions outside the contig are N-padded. Returns `NDArray[np.uint8]`.
- `contig_array(contig)` — the full contig sequence as a cached `NDArray[np.uint8]`. Shares the one-contig-in-memory cache with `fetch`. Accepts `chr`-prefixed or unprefixed names.
- Contig-name agnostic: `"chr1"` and `"1"` both resolve correctly (`ContigNormalizer` under the hood).
- One contig is cached in memory at a time; sequential per-contig access is efficient.

## Mutation catalogues (SBS-96 / DBS-78 / ID-83)

### `write_view` progress bar

`SparseVar.write_view(..., progress=False)` accepts an opt-in `progress` keyword.
When `True`, a phase-level `rich` progress bar is shown while the view is written
(one tick per major step: counting, genotypes, each carried field, the index
build, and mutation annotation when `reference=` is given). It defaults to
`False` — no bar and no overhead — so library and pipeline callers are
unaffected. The `genoray view` CLI exposes the same option as `--progress`
(also default off):

```bash
genoray view in.svar out.svar -r chr1:1-1000 -s A,B --progress
```

The bar is cosmetic: output bytes, schema, and dtypes are identical whether or
not it is enabled.

### Atomic crash-safe writes

Writes are crash-safe and atomic. `from_vcf`, `from_pgen`, and `write_view` build
the `.svar` directory in a hidden sibling staging directory (`.<name>.tmp…` next
to the output) and atomically rename it into place only after the write fully
succeeds; `.gvi` index files are written the same way. A crash mid-write never
leaves a partial or corrupt output, and overwriting an existing output preserves
it until the replacement is complete. Output bytes are unchanged — this is a
durability guarantee only.

### Overview

`SparseVar` supports COSMIC-style mutation catalogues. The workflow is:

1. Call `svar.annotate_mutations(reference)` once to classify every variant and
   write `mutcat.npy` to the `.svar` directory.
2. Call `svar.mutation_matrix(kind)` to get a per-sample count matrix.

### `SparseVar.annotate_mutations`

```python
svar = genoray.SparseVar("out.svar")
ref  = genoray.Reference.from_path("hg38.fa")

svar.annotate_mutations(ref)                   # write_back=True (default)
svar.annotate_mutations(ref, write_back=False) # in-memory only; not persisted
svar.annotate_mutations("hg38.fa")             # path accepted directly
```

Signature: `annotate_mutations(reference, *, contigs=None, write_back=True) -> None`

- `reference` — a `genoray.Reference` instance **or** a path to a FASTA file
  (auto-wraps via `Reference.from_path`).
- `contigs=None` — if given (a list of contig names), only variants on those
  contigs are classified; entries on all other contigs are marked
  `NOT_ANNOTATED` (sentinel `-4`) and their contigs are never fetched from the
  reference. Names match via the `ContigNormalizer` (`chr1`/`1` both work).
  Requested contigs absent from the `.svar` index are skipped with a warning; a
  listed contig present in the index but absent from the reference raises (omit
  it from the list to exclude it cleanly). `None` (default) classifies all
  contigs. When `write_back=True`, the normalized scope is recorded in
  `metadata.json` as `mutcat_contigs` (`None` = all).
- `write_back=True` — persists `mutcat.npy` and updates `metadata.json` so
  that subsequent `SparseVar(dir, fields=["mutcat"])` opens will see the field.
  **Note:** `write_view` **never** copies `mutcat` positionally to the output
  (see below); pass `reference=` to `write_view` to recompute it on the subset,
  or call `annotate_mutations` on the output view yourself.
- `write_back=False` — the `mutcat` field lives only in memory
  (`svar.fields["mutcat"]`); reopening the file will NOT find it.
- After the call, `svar.fields["mutcat"]` is populated regardless of
  `write_back`.

What it classifies:

| Variant type | Channel |
|---|---|
| Isolated SNV | SBS-96 (trinucleotide context) |
| Adjacent SNV pair on the same haplotype | DBS-78 (5' entry = DBS code, 3' entry = `DBS_PARTNER` sentinel) |
| Runs of ≥ 3 adjacent SNVs | SBS (each stays independent; no DBS collapse) |
| Native 2 bp MNV in the VCF | DBS-78 |
| MNV > 2 bp, symbolic, non-ACGT | UNCLASSIFIED |
| Insertion / deletion | ID-83 (size, repeat-context bucketing) |
| Variant on a contig outside `contigs=` | `NOT_ANNOTATED` (excluded from all matrices) |

### `SparseVar.mutation_matrix`

```python
svar = genoray.SparseVar("out.svar", fields=["mutcat"])  # pre-load field
df = svar.mutation_matrix("SBS96")                        # default count="allele"
df = svar.mutation_matrix("DBS78", count="sample")
df = svar.mutation_matrix("ID83",  count="allele")
```

Signature: `mutation_matrix(kind, *, count="allele") -> pl.DataFrame`

- `kind` — one of `"SBS96"`, `"DBS78"`, `"ID83"`.
- `count="allele"` — counts every non-ref allele copy (diploid homozygous = 2).
- `count="sample"` — counts each category at most once per sample (presence/absence).
- Returns a Polars `DataFrame` with a `MutationType` string column followed by
  one `Int64` column per sample. Rows are in fixed COSMIC codebook order
  (96 / 78 / 83 rows respectively).
- Requires the `mutcat` field to be available: either loaded at open time with
  `fields=["mutcat"]`, or already in memory from a prior `annotate_mutations`
  call, or present on disk from a prior `annotate_mutations(write_back=True)`.
  Raises `ValueError` if none of those hold.

### The `mutcat` field

`mutcat` is an `int16` field stored per genotype entry (same ragged layout as
`genos`). The int16 code space is:

| Range | Channel |
|---|---|
| `[0, 96)` | SBS-96 |
| `[96, 174)` | DBS-78 |
| `[174, 257)` | ID-83 |
| `-1` | `DBS_PARTNER` — 3' half of an adjacent SNV pair; never counted |
| `-2` | `UNCLASSIFIED` — symbolic / complex / MNV > 2 bp / non-ACGT |
| `-3` | `MISSING` — reserved sentinel (defined in the code space but not emitted by `annotate_mutations` v1; SparseVar stores only ALT-carrying entries, so no-call slots do not appear in the ragged field) |
| `-4` | `NOT_ANNOTATED` — entry on a contig outside the `contigs=` annotation scope; never counted |

To read a previously annotated file:

```python
svar = genoray.SparseVar("out.svar", fields=["mutcat"])
# svar.fields["mutcat"] is a Ragged[int16] mirroring svar.genos
```

### v1 scope limits (no strand-bias; calibrated against PCAWG/SigProfiler rules)

- No strand-bias separation (no SBS-192 / transcriptional strand).
- DBS collapse applies only to **isolated adjacent pairs** on the same
  haplotype. Runs of ≥ 3 adjacent SNVs stay as individual SBS entries.
- Indel channel (ID-83) bucketing follows PCAWG/SigProfiler published rules and
  is pinned by the unit tests in `tests/test_mutcat.py`. Cross-validation
  against SigProfilerMatrixGenerator is deferred (it is not a declared
  dependency).

### Signature refitting (COSMIC)

Decompose a catalogue into per-sample COSMIC signature activities.

```python
import genoray

ref = genoray.cosmic_signatures("SBS96")        # pooch-fetched + cached
cat = svar.mutation_matrix("SBS96")              # MutationType + sample cols
act = genoray.fit_signatures(cat, ref)           # activities + cosine_similarity

# convenience: mutation_matrix -> fit_signatures in one call
act = svar.assign_signatures("SBS96")                       # default COSMIC ref
act = svar.assign_signatures("SBS96", reference=ref, min_activity=0.01)
act = svar.assign_signatures("SBS96", reference="my_sigs.txt")  # TSV path
```

Signatures:
- `cosmic_signatures(kind, *, version="3.4", genome="GRCh38") -> pl.DataFrame`
  — fetches/caches the COSMIC reference set for `kind ∈ {"SBS96","DBS78","ID83"}`.
  Returns a `MutationType` column (canonical codebook order) + one column per
  signature. `genome` is ignored for `ID83`.
- `fit_signatures(catalogue, reference, *, max_delta=0.01, min_activity=0.005, n_jobs=1, backend="loky") -> pl.DataFrame`
  — sparse forward-selection refit (NNLS + cosine-guided add + min-activity
  prune). Aligns rows by joining on `MutationType` (raises `ValueError` if the
  catalogue has a type missing from the reference). Returns one row per sample:
  `Sample`, one Float column per signature (counts; `0.0` if unselected), and
  `cosine_similarity`. `n_jobs=1` (default) is serial; `n_jobs=-1` uses all
  cores. Results are identical regardless of `n_jobs`/`backend`.
- `SparseVar.assign_signatures(kind, *, reference=None, count="allele", max_delta=0.01, min_activity=0.005, n_jobs=1, backend="loky") -> pl.DataFrame`
  — `mutation_matrix(kind, count=...)` then `fit_signatures(...)`. `reference`
  accepts a `pl.DataFrame`, a TSV path, or `None` (defaults to `cosmic_signatures(kind)`).
  Forwards `n_jobs`/`backend` to `fit_signatures` for per-sample parallelism
  (`n_jobs=1` (default) is serial; `n_jobs=-1` uses all cores).

Out of scope (v1): de novo extraction, opportunity normalization, bootstrap CIs,
plotting.

## Common mistakes

| Mistake | Fix |
|---|---|
| `genoray.Genos8` | `genoray.VCF.Genos8` (class attribute) |
| `vcf.read(..., phasing=True)` | Set `phasing=True` on the `VCF()` constructor |
| Reading dosages from a VCF without `dosage_field=` | Pass `dosage_field="DS"` (or appropriate `Number=A` field) on the constructor |
| Putting a dosage-only PGEN in the main path when you also have hardcalls | Hardcalls in main path, dosages in `dosage_path=` |
| Importing `from genoray._vcf import VCF` | Use `from genoray import VCF` |
| Expecting VCF to have `read_ranges` | VCF doesn't; loop over single-range `read` calls, or use PGEN/SparseVar |
| Treating `svar.index["POS"]` as 0-based | It's 1-based; subtract 1 to compare with query coords |
| Calling `read_ranges` and assuming a flat array | PGEN returns `(data, offsets)`; SparseVar returns a Ragged (or awkward record with `fields`) |
| Calling `mutation_matrix` without a `mutcat` field | Run `annotate_mutations` first, or open with `fields=["mutcat"]` |
| Expecting `mutation_matrix` to auto-run annotation | It does not; call `annotate_mutations` separately |
| Re-opening SparseVar and losing the `mutcat` field | Use `write_back=True` (default) in `annotate_mutations`; then open with `SparseVar(dir, fields=["mutcat"])` |
| Calling `write_view` and expecting `mutcat` to be in the output | `write_view` **never** copies `mutcat` positionally (subsetting invalidates DBS adjacency codes). Pass `reference=` to `write_view` to recompute it on the subset, or call `annotate_mutations` on the output view yourself. Explicitly including `"mutcat"` in `fields=` without a `reference=` raises `ValueError`. |
| Passing the source dataset directory as `output` to `write_view` (even with `overwrite=True`) | Raises `ValueError` — writing in place would delete the source before the view is written. Pass a different `output` path. |
| Passing a FASTA path directly to `annotate_mutations` | Supported — it auto-wraps via `Reference.from_path` |

## When this skill needs updating

Any PR that adds, removes, renames, or changes the semantics of a public
name (anything reachable from `import genoray` without underscores) must
update this skill alongside the code change. See the project `CLAUDE.md`.

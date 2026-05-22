---
name: genoray-api
description: Use when writing or modifying Python code that imports `genoray` to read genotypes/dosages from VCF, PGEN, or SparseVar (`.svar`) files. Covers the public API surface, mode constants, range queries, chunking, filtering, and the SparseVar workflow. Skip for unrelated bioinformatics work.
---

# genoray public API

`genoray` is a NumPy-first range-query layer over VCF/BCF (cyvcf2), PGEN
(pgenlib), and a sparse memmap format (`SparseVar` / `.svar`).

## Public surface

`import genoray` exposes exactly:

- `genoray.VCF` — VCF/BCF reader
- `genoray.PGEN` — PLINK 2 PGEN reader
- `genoray.SparseVar` — sparse `.svar` reader/writer
- `genoray.Reader` — type alias `VCF | PGEN | SparseVar`
- `genoray.exprs` — polars filter expressions for `.gvi` indexes

Nothing else is public. Anything starting with `_` (e.g. `genoray._vcf`) is
internal — do not import it from user code.

## Where to look for details

Prefer reading these over guessing:

- `docs/source/index.md` — narrative tour with full examples (VCF, PGEN, filtering, chunking)
- `docs/source/svar.md` — SparseVar usage
- `genoray/__init__.py` — confirms the public surface
- `genoray/_vcf.py` — `VCF` class: constructor, `read`, `chunk`, mode constants near the top of the class
- `genoray/_pgen.py` — `PGEN` class: constructor, `read`, `chunk`, `read_ranges`, `chunk_ranges`, mode constants near the top of the class
- `genoray/_svar.py` — `SparseVar`: `__init__`, `from_vcf`, `from_pgen`, `read_ranges`, `with_fields`
- `genoray/exprs.py` — the *complete* set of pre-built filter expressions (currently 4: `is_snp`, `is_indel`, `is_biallelic`, `ILEN`)

When a signature, kwarg, or shape is unclear, **read the docstring in the
source** rather than reasoning from first principles.

## Cross-cutting conventions

- Ranges are 0-based, half-open `[start, end)`.
- `max_mem` accepts strings like `"4g"`, `"512m"`, `"2GB"`.
- Contig names auto-normalize: `"chr1"` and `"1"` both work regardless of file convention (`ContigNormalizer`).
- Missing genotype = `-1` (int). Missing dosage = `np.nan` (float32).
- Ploidy is always 2.
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
    phasing=True,           # constructor-time, not per-read
    dosage_field="DS",      # required to read dosages; FORMAT field with Number=A
    filter=lambda v: ...,   # cyvcf2.Variant -> bool
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
```

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

VCF: pass a `Callable[[cyvcf2.Variant], bool]` to `filter=`.

PGEN: pass a polars `pl.Expr` returning a boolean mask, operating on the
`.gvi` index columns. Built-in expressions in `genoray.exprs` (the
*complete* list):

- `is_snp`
- `is_indel`
- `is_biallelic`
- `ILEN` (an expression yielding indel length, not a boolean)

For anything else, write `pl.col(...)` against the `.gvi` schema — read
`genoray/exprs.py` for the available columns. Combining two `exprs`
expressions with `&` / `|` works without importing polars; you only need
`import polars as pl` to build custom predicates.

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

## When this skill needs updating

Any PR that adds, removes, renames, or changes the semantics of a public
name (anything reachable from `import genoray` without underscores) must
update this skill alongside the code change. See the project `CLAUDE.md`.

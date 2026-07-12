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
- `genoray.Filter` — VCF filter value object bundling a cyvcf2 record predicate (`record`) with its matching `.gvi` polars expression (`expr`)
- `genoray.SparseVar` — sparse `.svar` reader/writer
- `genoray.SparseVar2` — next-gen sparse variant store (VCF/BCF → SVAR2 conversion via `from_vcf`; range queries via `decode`/`region_counts`/`read_ranges`; mutational-signature support (SBS96/DBS78/ID83) via `annotate_mutations`/`mutation_matrix`/`assign_signatures`, or classify during the write with `from_vcf(signatures=True)`; scalar-numeric INFO/FORMAT field extraction during the write via `from_vcf(info_fields=, format_fields=)`)
- `genoray.InfoField` / `genoray.FormatField` — frozen dataclasses (`name`, `dtype=None`, `default=None`) configuring a single INFO/FORMAT field for `SparseVar2.from_vcf`; a bare `str` name uses inferred defaults instead
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
- `genoray/_vcf.py` — `VCF` class: constructor, `read`, `chunk`, mode constants near the top of the class; `get_record_info(contig=None, start=None, end=None, fields=None, info=None, lazy=False)` — non-FORMAT record-level fields (including INFO) for a range or the whole file, returns `pl.DataFrame` (or `pl.LazyFrame` when `lazy=True`)
- `genoray/_pgen.py` — `PGEN` class: constructor, `read`, `chunk`, `read_ranges`, `chunk_ranges`, mode constants near the top of the class
- `genoray/_svar.py` — `SparseVar`: `__init__`, `from_vcf`, `from_pgen`, `read_ranges`, `read_ranges_with_length(contig, starts=0, ends=POS_MAX, samples=None)` (length-guaranteed range read; returns the same type as `read_ranges` — a `Ragged` or fields-augmented record), `with_fields`, `annotate_mutations`, `mutation_matrix`, `assign_signatures`, `annotate_with_gtf(gtf, level_filter=1, write_back=True, *, strand_encoding=None, codon_null_token=None)` (GTF CDS annotation entry point, returns `pl.DataFrame` with `varID`/`gene_id`/`strand`/`codon_pos`), `cache_afs()` (computes and persists an `AF` column to the `.gvi` index; returns `None`)
- `genoray/_svar2.py` — `SparseVar2`: `__init__`, `from_vcf` (VCF/BCF → SVAR2 conversion entry point, `signatures=` classifies during the write, `info_fields=`/`format_fields=` extract scalar-numeric fields during the write); `n_samples`/`available_samples`/`contigs`/`ploidy` metadata. Read/query methods live in the mixins: `genoray/_svar2_decode.py` (`decode`, `region_counts`), `genoray/_svar2_batch.py` (public `read_ranges`; internal gvl-only `_overlap_batch`/`_find_ranges`/`_gather_ranges`), and `genoray/_svar2_mutcat.py` (`annotate_mutations`, `mutation_matrix`, `assign_signatures` — COSMIC mutational-signature workflow, mirroring `SparseVar`'s but backed by a per-contig Rust sidecar instead of a `.gvi`-attached field)
- `genoray/_svar2_fields.py` — `InfoField`/`FormatField` dataclasses + `FieldDtype` and the header/dtype validation used by `from_vcf(info_fields=, format_fields=)` (no public read-side API yet)
- `genoray/_cli/__main__.py` — the `genoray` CLI (`index`, `write` / `write svar1`, `view`)
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
- Ploidy is 2 by default; `SparseVar.from_vcf`/`from_pgen` (and `genoray write svar1`) accept `haploid=True` / `--haploid`, which OR-collapses haplotypes into a single haploid call per sample and records `ploidy=1` in metadata (intended for unphased somatic data).
- All return arrays are NumPy; `mode` selects which arrays you get back.

## Sample accessors — canonical name + why the idioms diverge

`available_samples` (a `list[str]`) is the canonical "all samples in the
file" accessor — present on all four readers (`VCF`, `PGEN`, `SparseVar`,
`SparseVar2`).

`VCF` and `PGEN` additionally expose:
- `current_samples` — the currently-selected subset (read-only property).
- `set_samples(samples) -> Self` — a stateful call that mutates the reader
  in place to select a subset (or restore all samples with `None`), then
  returns `self`.

`SparseVar` and `SparseVar2` have **no** `current_samples`/`set_samples`.
Instead, every read method (`read_ranges`, `read_ranges_with_length`, etc.)
takes samples as a per-call `samples=` kwarg.

**Why the two idioms differ (performance):** subsetting samples on VCF/PGEN
is costly — it re-initializes the backend reader — so it's a deliberate,
stateful `set_samples()` call made once and reused across reads. On
`SparseVar`/`SparseVar2`, subsetting is ~free (it's just an index selection
over already-memory-mapped data), so it's exposed as a lightweight per-call
`samples=` kwarg instead of a persistent reader state. This is an
intentional divergence, not an inconsistency — don't "fix" one to match
the other.

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
    filter=genoray.Filter(
        record=lambda v: ...,                 # cyvcf2.Variant -> bool
        expr=~genoray.exprs.is_symbolic,       # matching .gvi index predicate
    ),
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
- `read(out=...)` is **VCF-only** — pass a pre-allocated array to fill in place. PGEN random-access reads allocate fresh and have no `out=` buffer.

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

## SparseVar2 (`.svar2`) — quick reference

`SparseVar2` is the next-gen sparse variant store (VariantKey-style inline
encoding + per-variant dense/sparse cost model). Two halves: **conversion**
(`from_vcf`, below) writes a store; **range queries** (`decode` / `region_counts`
/ `read_ranges`, further below) read it back. All coordinates are 0-based
half-open `[start, end)`, as everywhere else in genoray.

### Conversion

```python
from genoray import SparseVar2

dropped = SparseVar2.from_vcf(
    "out.svar2", "file.vcf.gz", "ref.fa",   # reference: validates REF + left-aligns indels
    overwrite=True,
)

# Pre-normalized input (e.g. `bcftools norm`'d): skip REF validation/left-align
dropped = SparseVar2.from_vcf("out.svar2", "file.vcf.gz", no_reference=True)
```

Signature: `from_vcf(out, source, reference=None, *, no_reference=False, skip_out_of_scope=False, ploidy=2, chunk_size=25_000, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False, info_fields=None, format_fields=None) -> int`

- `source` — a bgzipped VCF (`.vcf.gz`) or BCF (`.bcf`). **VCF/BCF only** — no PGEN
  source yet (roadmap M7). Auto-indexes (`.csi`) if no `.csi`/`.tbi` is found.
- Exactly one of `reference` (a FASTA path, used to validate REF and left-align
  indels) or `no_reference=True` (trusts pre-normalized input, skips
  validation/left-align) is required — passing both or neither raises
  `ValueError`.
- `skip_out_of_scope=False` — when `True`, drops out-of-scope (symbolic
  `<DEL>`/`<INS>`/… and breakend) ALTs instead of erroring; the strict default
  errors on the first one. The two classes are **not** distinguishable at this
  layer — there's no separate "symbolic only" vs. "breakend only" toggle.
- Returns the number of dropped out-of-scope ALTs as an `int` (always `0`
  unless `skip_out_of_scope=True`).
- No dosages, no `haploid=` OR-collapse, no `max_mem`-based chunking (use
  `chunk_size` instead) — these remain `SparseVar` (SVAR 1.0)-only for now.
- `signatures=False` — when `True`, classifies every SNP/indel into its
  SBS96/ID83 mutation-type code during the write and stores a `mutcat`
  sidecar per contig (factored into the write's dense/var_key cost model).
  Requires a reference (`reference=`); raises `ValueError` if combined with
  `no_reference=True`. There is no public read-side API for the SVAR2
  `mutcat` sidecar yet (unlike `SparseVar.annotate_mutations`/
  `mutation_matrix`, below) — this flag only controls whether the sidecar is
  written.
- `info_fields=`/`format_fields=` — `Sequence[str | InfoField]` /
  `Sequence[str | FormatField]`, `None` by default. Extracts **scalar-numeric**
  INFO/FORMAT fields into the store during the write:

  ```python
  from genoray import SparseVar2, InfoField, FormatField

  SparseVar2.from_vcf(
      "out.svar2", "file.vcf.gz", "ref.fa",
      info_fields=["AC", InfoField("AF", dtype="f16")],
      format_fields=[FormatField("DS", default=0.0)],
  )
  ```

  - **Scope: scalar-numeric only.** Header `Type` must be `Integer`, `Float`,
    or `Flag`; `Number` must be `1`, biallelic-split `A`, or `0` (`Flag`,
    INFO-only). Anything else (`Number=R`/`G`/`.`, `String`/`Character`
    fields) raises `ValueError` at config time, before conversion starts. A
    bare `str` name uses inferred defaults (`dtype=None`, no `default`); pass
    an `InfoField`/`FormatField` to override.
  - **`dtype`** (`FieldDtype = Literal["bool","i8","u8","i16","u16","i32","u32","f16","f32"]`):
    `None` (default) auto-resolves — `Integer`/`Flag` are **losslessly
    auto-narrowed** to the smallest width fitting the observed global range
    (plus a reserved missing sentinel); `Float` always resolves to `f32`
    (never silently downcast). An explicit `dtype` is validated at
    conversion time against both the header type (e.g. `Float` cannot target
    an int width) and the observed range — overflow, or `f16`'s ~65504
    range, raises `ValueError`. `f16` is the only lossy option and must be
    requested explicitly.
  - **`default`** — the value written for VCF-missing entries; otherwise a
    reserved sentinel (`NaN` for float widths, the width-specific
    `INT*_MIN` for ints). `Flag` fields are never missing (absent ⇒
    `false`/`0`).
  - **FORMAT is genotype-aligned, not independently lossless**: a FORMAT
    value is stored only where the genotype has a call — one value per
    carrier call in var_key-routed variants, or a full dense per-sample
    column (non-carrier slots filled with `default`/sentinel) in
    dense-routed variants. Non-carrier FORMAT values (e.g. an imputed
    dosage at a ref/ref genotype) are **dropped by design** in this version;
    an independent lossless FORMAT stream is deferred to a future spec.
  - **No read path yet** — this only controls what gets written. Querying
    stored field values back out (a decode/query API) is not implemented;
    see `meta.json`'s `"fields"` array and `docs/roadmap/data-model.md` for
    the on-disk layout if you need to inspect values directly.

### Range queries

Open a finished store, then query per contig. Construction reads `meta.json` and
opens one native reader per contig, exposing `.available_samples` (list; the
canonical sample-name accessor shared with `VCF`/`PGEN`/`SparseVar`),
`.n_samples`, `.contigs`, `.ploidy`, `.format_version`.

```python
from genoray import SparseVar2

sv = SparseVar2("out.svar2")
regions = [(0, 40), (1_000, 2_000)]   # 0-based half-open [start, end)

# Analysis path — decode to a seqpro Ragged record (one call per contig)
rag = sv.decode("chr1", regions)      # fields pos (i32), ilen (i32), allele (ALT bytes)
                                      # shape (R, S, P, None); pure-DEL ALT is empty

# Decode-free per-(region, sample, ploid) variant count — replaces SVAR 1.0's var_ranges
counts = sv.region_counts("chr1", regions)   # np.ndarray, shape (R, S, P)
```

- `decode(contig, regions)` returns a `seqpro.rag.Ragged` whose layout is
  **byte-identical to gvl's `RaggedVariants`** (`pos`/`ilen` numeric,
  `allele` opaque-string ALT, one shared variant-axis offsets object). ALT is
  empty for a pure deletion (the reference base is not re-emitted). Requires
  `seqpro`.
- `region_counts(contig, regions)` is the **decode-free** count (offset diffs +
  dense-mask popcount) — the simplified stand-in for `SparseVar.var_ranges`
  (SVAR2 has no unified variant table, so variant *indices* no longer exist).
- Queries are **per contig** — cross-contig batching is the caller's job. Regions
  are an iterable of `(start, end)` pairs.

The user-facing SVAR2 query API is `decode` / `region_counts` / `read_ranges`
(above). `read_ranges(contig, starts, ends, samples=None)` is a fused
search+gather; `starts`/`ends` are parallel 1D arrays (mirrors
`SparseVar.read_ranges`), `samples` selects/reorders a subset **by name**. It
returns the raw two-channel `BatchResult` → numpy dict, a `TypedDict` with a
fixed field set: `vk_pos`/`vk_key`/`vk_off`, `dense_pos`/`dense_key`/
`dense_range`/`dense_present`/`dense_present_off`, `lut_bytes`/`lut_off`, and
scalars `n_regions`/`n_samples`/`ploidy`.

`SparseVar2` also has `_overlap_batch`/`_find_ranges`/`_gather_ranges`
(underscore-prefixed) — an internal, gvl-only numpy-dict wire contract for
the search/gather split used by a write-time overlap cache. They are **not**
part of the public API, are not covered by semver, and may change or
disappear without notice; don't call them from user code.

### Mutational signatures (SBS96 / DBS78 / ID83)

Same COSMIC workflow as `SparseVar` (see "Mutation catalogues" above), backed
by a Rust per-contig sidecar instead of a `.gvi`-attached field. Annotation is
**required** before `mutation_matrix` — either post-hoc, or by passing
`signatures=True` to `from_vcf` (above):

```python
sv = SparseVar2("out.svar2")
ref = genoray.Reference.from_path("hg38.fa")

sv.annotate_mutations(ref)                   # post-hoc; writes the mutcat sidecar
sv.annotate_mutations(ref, contigs=["chr1"])  # restrict to a subset of contigs

df = sv.mutation_matrix("SBS96")                     # count="allele" (default)
df = sv.mutation_matrix("DBS78", count="sample")
act = sv.assign_signatures("SBS96")                  # mutation_matrix + fit_signatures
```

- `annotate_mutations(reference, *, contigs=None) -> None` — `reference` is a
  `genoray.Reference` or a FASTA path; `contigs=None` (default) annotates every
  contig. Unlike `SparseVar.annotate_mutations`, there is **no `write_back=`
  toggle** — SVAR2 always persists the sidecar to disk.
- `mutation_matrix(kind, *, count="allele"|"sample") -> pl.DataFrame` — a
  `MutationType` column (fixed COSMIC codebook order) plus one column per
  sample. `kind ∈ {"SBS96", "DBS78", "ID83"}`. `count="allele"` counts every
  non-ref allele copy; `count="sample"` counts each category at most once per
  sample, OR-combined across contigs. **Raises `ValueError`** if called before
  the store is annotated (no on-disk sidecar for every contig) — annotate
  first, either via `annotate_mutations` or `from_vcf(..., signatures=True)`.
- `assign_signatures(kind, *, reference=None, count="allele", max_delta=0.01, min_activity=0.005, n_jobs=1, backend="loky") -> pl.DataFrame`
  — `mutation_matrix(kind, count=...)` then `genoray.fit_signatures(...)`.
  `reference` accepts a `pl.DataFrame`, a TSV path, or `None` (defaults to
  `genoray.cosmic_signatures(kind)`).
- Same classification rules as v1 (shared Rust classifier): **DBS78 arises
  only from isolated adjacent same-haplotype SNV pairs** — runs of ≥3
  adjacent SNVs stay as individual SBS96 entries, native MNVs > 2bp are
  atomized into SBS96, and each isolated doublet is counted **once** (not
  once per constituent SNV).
- No public read-side access to the raw per-genotype `mutcat` codes for
  SVAR2 (unlike v1's `fields=["mutcat"]`) — only the aggregated
  `mutation_matrix` output is exposed.

### Errors

genoray raises standard Python builtins, by category:

- `ValueError` — bad input content: contig/sample not found, REF disagrees with
  the reference FASTA, or a symbolic/breakend ALT with `skip_out_of_scope=False`.
- `FileNotFoundError` — a required input file is missing.
- `OSError` — a corrupt/truncated store sidecar or an underlying disk I/O failure.
- `RuntimeError` — an internal genoray bug (a worker thread panicked); please
  report it.

## CLI (`genoray write`)

`genoray write` **defaults to SVAR2**; `genoray write svar1` runs the previous
SVAR 1.0 behavior.

```bash
# SVAR2 (default) — reference required XOR --no-reference
genoray write file.vcf.gz out.svar2 --reference ref.fa
genoray write file.vcf.gz out.svar2 --no-reference
genoray write file.vcf.gz out.svar2 --reference ref.fa --skip-symbolics-and-breakends --threads 4

# SVAR 1.0 (previous default) — VCF or PGEN, dosages, --haploid, --max-mem
genoray write svar1 file.vcf.gz out.svar --max-mem 4g --haploid
```

- `genoray write` (SVAR2, thin wrapper over `SparseVar2.from_vcf`): `--reference`
  XOR `--no-reference` (required), `--ploidy` (default 2), `--chunk-size`
  (default 25000), `--threads`/`-@`, `--overwrite`, `--long-allele-capacity`
  (advanced), and a single `--skip-symbolics-and-breakends` flag (maps to
  `skip_out_of_scope=`) — the SVAR2 core can't expand either symbolic ALTs
  (`<DEL>`, `<INS>`, …) or breakends into nucleotides, so they're dropped
  together; prints a `Dropped {n} out-of-scope (symbolic/breakend) ALT
  alleles.` line when set. VCF/BCF source only.
- `genoray write svar1`: unchanged SVAR 1.0 behavior — VCF or PGEN source,
  `--dosages`, `--max-mem`, `--haploid`, `--no-symbolic`/`--no-breakend`
  (independent flags here, unlike SVAR2).

## Filtering

VCF: pass a `genoray.Filter(record=, expr=)` value object to `filter=`.
`record` is a `Callable[[cyvcf2.Variant], bool]` applied during the genotype
scan; `expr` is the matching polars `pl.Expr` applied to the `.gvi` index —
VCF requires **both** halves, bundled together so they can never diverge.

To change a VCF's filter after construction, assign a `Filter` (or `None` to
clear it) to the `vcf.filter` setter; the in-memory index is invalidated.
The getter returns the `Filter | None` currently in effect, so `vcf.filter =
vcf.filter` round-trips.

```python
from genoray import VCF, Filter

vcf = VCF("file.vcf", filter=Filter(
    record=lambda v: not v.INFO.get("SVTYPE"),   # cyvcf2 record predicate
    expr=~genoray.exprs.is_symbolic,               # matching .gvi index predicate
))
vcf.filter = None                                  # clear
f = vcf.filter                                      # -> Filter | None
```

The former two-argument constructor (a separate polars-expression keyword
argument alongside `filter=`) and its tuple-valued `vcf.filter` getter/setter
are **removed in 3.0.0** — migrate any code passing the record predicate and
polars expression separately to the single `Filter(record=, expr=)` object
shown above.

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

**Filtering guidance** (use `filter=` — a bare `pl.Expr` for PGEN, a
`genoray.Filter` for VCF):

- `~genoray.exprs.is_symbolic` — drops *all* symbolic alleles (precise or not).
  Required for haplotype consumers (e.g. `genvarloader`) that cannot expand any
  symbolic ALT into literal sequence:

  ```python
  # PGEN
  pgen = genoray.PGEN("file.pgen", filter=~genoray.exprs.is_symbolic)
  # VCF (both halves required, bundled in a Filter)
  vcf = genoray.VCF(
      "file.vcf.gz",
      filter=genoray.Filter(
          record=lambda rec: not any(a.startswith("<") for a in rec.ALT),
          expr=~genoray.exprs.is_symbolic,
      ),
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

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
- `genoray.SparseVar2` — next-gen sparse variant store (VCF/BCF → SVAR2 conversion via `from_vcf`, PLINK2 PGEN → SVAR2 conversion via `from_pgen`, N single-sample VCFs/BCFs → one SVAR2 store via a native k-way merge in `from_vcf_list` (`reference`/`no_reference` supported like `from_vcf`, absent sites fill hom-ref), SVAR1 (`SparseVar`) → SVAR2 native migration via `from_svar1` (reads no VCF/htslib; biallelic SVAR1 only; supports `regions=`/`samples=`/`merge_overlapping=`/`regions_overlap=` like `from_vcf`/`from_pgen`); range queries via `decode`/`region_counts`/`read_ranges`; mutational-signature support (SBS96/DBS78/ID83) via `annotate_mutations`/`mutation_matrix`/`assign_signatures`, or classify during the write with `from_vcf(signatures=True)`/`from_pgen(signatures=True)`/`from_svar1(signatures=True)`; scalar-numeric INFO/FORMAT field extraction during the write via `from_vcf(info_fields=, format_fields=)`/`from_vcf_list(info_fields=, format_fields=)` (`from_vcf_list` merges INFO first-carrier-wins, FORMAT per-sample) — not supported by `from_pgen`/`from_svar1` (which carries through all of SVAR1's existing fields automatically) — read back opt-in via `fields=`/`with_fields`/`available_fields` and attached to `decode`'s result)
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
- `genoray/_svar2.py` — `SparseVar2`: `__init__(path, *, fields=None)`, `with_fields(fields)` (new reader over the same store with those fields selected), `available_fields` (`dict[str, StoredField]`, set in `__init__`), `from_vcf` (VCF/BCF → SVAR2 conversion entry point, `signatures=` classifies during the write, `info_fields=`/`format_fields=` extract scalar-numeric fields during the write), `from_pgen` (PLINK2 PGEN → SVAR2 conversion entry point; diploid-only, no `ploidy=`/`info_fields=`/`format_fields=`; supports `regions=`/`samples=`/`merge_overlapping=`/`regions_overlap=` like `from_vcf`), `from_vcf_list` (N single-sample VCFs/BCFs → one SVAR2 store via a native k-way merge; `sources` accepts a `Sequence`/directory/manifest, resolved by module-level `_resolve_vcf_sources`; `reference`/`no_reference` supported (no_reference skips left-alignment, so cross-file joins require pre-normalized inputs); `info_fields=`/`format_fields=` supported — INFO merges first-carrier-wins, FORMAT stays per-sample), `from_svar1` (SVAR1 (`SparseVar`) → SVAR2 native migration entry point; reads no VCF/htslib, `ploidy` from SVAR1 metadata, biallelic SVAR1 only, no `info_fields=`/`format_fields=` — every SVAR1 FORMAT field carries through automatically, `mutcat` dropped; supports `regions=`/`samples=`/`merge_overlapping=`/`regions_overlap=` like `from_vcf`/`from_pgen`, though regions filter per-record rather than narrowing a covering range up front); `n_samples`/`available_samples`/`contigs`/`ploidy` metadata. Read/query methods live in the mixins: `genoray/_svar2_decode.py` (`decode` — attaches one `Ragged` per selected field, `region_counts`), `genoray/_svar2_batch.py` (public `read_ranges`; internal gvl-only `_overlap_batch`/`_find_ranges`/`_gather_ranges`), and `genoray/_svar2_mutcat.py` (`annotate_mutations`, `mutation_matrix`, `assign_signatures` — COSMIC mutational-signature workflow, mirroring `SparseVar`'s but backed by a per-contig Rust sidecar instead of a `.gvi`-attached field)
- `genoray/_svar2_fields.py` — `InfoField`/`FormatField` dataclasses + `FieldDtype` and the header/dtype validation used by `from_vcf(info_fields=, format_fields=)`; `StoredField` (frozen dataclass: `name`, `category`, `dtype`, `default`, `key`) is the read-side manifest entry type returned by `SparseVar2.available_fields` — not exported at top-level `genoray`, only reached via that dict
- `genoray/_cli/__main__.py` — the `genoray` CLI (`index`, `write` / `write svar1`, `view` / `view svar1`, `concat`, `split`)
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

Signature: `from_vcf(out, source, reference=None, *, no_reference=False, skip_out_of_scope=False, ploidy=2, chunk_size=25_000, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False, info_fields=None, format_fields=None, check_ref="e") -> int`

- `source` — a bgzipped VCF (`.vcf.gz`) or BCF (`.bcf`). Auto-indexes (`.csi`) if
  no `.csi`/`.tbi` is found. For a PLINK2 PGEN source, use `from_pgen` instead
  (below).
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
- `check_ref: Literal["e", "x"] = "e"` — policy for a record whose REF
  disagrees with the reference FASTA (ignored when `no_reference=True`).
  `"e"` (default) raises and aborts the build, matching `bcftools norm
  --check-ref e`. `"x"` drops the offending record (including a REF that
  runs past the contig end) and continues, logging a per-contig count.
  Comparison is case-insensitive (soft-masked lowercase reference bases
  match). Any other value raises `ValueError` before conversion starts.
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
    reserved sentinel at the extreme of the chosen width (`INT*_MIN` for
    signed widths, `u*::MAX` for unsigned widths — auto-narrowing prefers
    unsigned when the observed range is non-negative — and `NaN` for float
    widths). `Flag` fields are never missing (absent ⇒ `false`/`0`).
  - **FORMAT is genotype-aligned, not independently lossless**: a FORMAT
    value is stored only where the genotype has a call — one value per
    carrier call in var_key-routed variants, or a full dense per-sample
    column (non-carrier slots filled with `default`/sentinel) in
    dense-routed variants. Non-carrier FORMAT values (e.g. an imputed
    dosage at a ref/ref genotype) are **dropped by design** in this version;
    an independent lossless FORMAT stream is deferred to a future spec.
  - **Read path:** see "Reading INFO/FORMAT fields (SVAR2)" below —
    `SparseVar2(path, fields=…)` / `.with_fields(…)` / `.available_fields`
    opt into decoding these back out via `decode()`.

### Conversion from PGEN

```python
from genoray import SparseVar2

dropped = SparseVar2.from_pgen(
    "out.svar2", "file.pgen", "ref.fa",   # reference: validates REF + left-aligns indels
    overwrite=True,
)
```

Signature: `from_pgen(out, source, reference=None, *, regions=None, samples=None, merge_overlapping=False, regions_overlap="pos", no_reference=False, skip_out_of_scope=False, chunk_size=None, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False, check_ref="e") -> int`

- `source` — a `.pgen` file. Variant metadata is read from the sibling
  `.pvar`/`.pvar.zst`, sample names from the sibling `.psam`.
  `reference`/`no_reference`, `skip_out_of_scope`, `overwrite`,
  `long_allele_capacity`, `signatures`, and `check_ref` all mean the same as
  `from_vcf` (above), and return the same `int` (dropped out-of-scope ALTs).
- **Diploid only** — no `ploidy=` kwarg (`from_vcf`'s default `ploidy=2` is
  implicit and fixed here).
- `chunk_size=None` — unlike `from_vcf`'s fixed `25_000` default, `None` here
  derives a variant-count budget from sample count (a packed dense chunk costs
  `chunk_size * n_samples * 2 / 8` bytes), so a fixed constant that's fine at
  200 samples doesn't blow memory at 500k. Pass an explicit `int` to override.
- **`regions=`/`merge_overlapping=`/`regions_overlap=`** — same convention,
  semantics, and three overlap modes (`"pos"`/`"record"`/`"variant"`) as
  `from_vcf`, restricting conversion to one or more `.pvar` variant-index
  ranges. As with `from_vcf`, `"variant"` mode keeps a multiallelic record
  whole if ANY of its alleles truly overlaps the region.
- **`samples=`** — selects and reorders `.psam` samples by name (same
  convention as `from_vcf`): preserves caller order, de-duplicates first
  occurrences, raises `ValueError` on an unknown name. `available_samples`
  and every decoded column match the caller's order exactly, regardless of
  each sample's original `.psam` position.
- **No `info_fields=`/`format_fields=`** — PGEN carries no FORMAT, and `.pvar`
  INFO extraction is not implemented.
- **No dosages** — a `.pgen` dosage track (if present) is ignored; only
  hardcalls are read.
- Unphased heterozygotes resolve haplotypes in the allele-code order
  `pgenlib` returns — the same caveat `from_vcf` carries for unphased `GT`.

### Conversion from a list of single-sample VCFs

```python
from genoray import SparseVar2

# Explicit list
dropped = SparseVar2.from_vcf_list("out.svar2", ["s1.vcf.gz", "s2.bcf"], "ref.fa")

# A directory of single-sample files (non-recursive: all *.vcf.gz, then all *.bcf)
dropped = SparseVar2.from_vcf_list("out.svar2", "vcfs/", "ref.fa")

# A manifest file (one path per line; blank/`#`-comment lines skipped;
# relative entries resolved against the manifest's directory)
dropped = SparseVar2.from_vcf_list("out.svar2", "manifest.txt", "ref.fa")
```

Signature: `from_vcf_list(out, sources, reference=None, *, no_reference=False, skip_out_of_scope=False, ploidy=2, chunk_size=25_000, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False, info_fields=None, format_fields=None, check_ref="e") -> int`

Builds **one** SVAR2 store from **N single-sample** VCFs/BCFs with different
site lists, via a native k-way merge — no `bcftools merge`, no intermediate
multi-sample VCF.

- **Each input file must be single-sample** — exactly one sample column;
  `ValueError` if any file has zero or more than one. That sample's VCF
  header name becomes its sample name in the store; duplicate sample names
  across input files raise `ValueError`.
- `sources` — one of three forms, resolved by module-level
  `_resolve_vcf_sources`:
  - a `Sequence[str | Path]` — explicit files, in the given order.
  - a single directory `Path` — every `*.vcf.gz` then every `*.bcf` directly
    inside it (non-recursive), each group `natsort`-ordered.
  - a single file `Path` — `.vcf.gz`/`.bcf` is taken as one file; anything
    else is a manifest (one path per line, blank/`#`-comment lines skipped,
    relative entries resolved against the manifest's parent directory).
  - Resolving to zero files raises `ValueError`.
- **Absent site → hom-ref `0`.** A site called in file A but not present at
  all in file B fills `0` (hom-ref) for B's sample at that site.
- **A within-file `./.` is not observable after the merge.** SVAR2's sparse
  layout stores only ALT-carrying entries, so a missing hap and a hom-ref hap
  both produce zero entries and cannot be told apart via `decode` or
  `region_counts`. The `-1` missing sentinel is a dense
  `genoray.VCF`/`genoray.PGEN` convention and is **not** part of SVAR2's
  decode. (The distinction is real inside the merge, but it is discarded when
  genotypes are packed into the sparse carrier bit-grid — this matches
  `from_vcf`, so the two paths stay in parity.)
- The merge is join-on-atom — a variant is one shared row across files iff its
  normalized `(pos, ref, alt)` atom matches exactly, not merely its position.
- **Each input file's records must already be position-sorted** per contig
  (same assumption `from_vcf` makes for its single input) — an unsorted
  file raises `ValueError` naming the offending file and positions rather
  than silently corrupting the k-way merge.
- **Every input file must use the same contig naming scheme** (all
  `chr1`-style or all `1`-style, not a mix) — the merge matches contigs by
  an exact per-file string, so a cohort mixing schemes raises `ValueError`
  up front (naming the conflicting files/spellings) instead of silently
  producing a store where half the cohort's samples decode as all-zeros on
  the "wrong-spelled" contigs.
- **Opens all N input files concurrently** (one file descriptor per file per
  contig) — at large N (roughly `N > (soft RLIMIT_NOFILE - 64) / 2`, often
  around N ≈ 480 at a default 1024 soft limit) this raises `ValueError` with
  the `ulimit -n` remedy instead of htslib's more confusing "is there a .tbi
  or .csi file?" error for some arbitrary file near the ceiling. There is no
  batched/hierarchical merge to fall back on for very large cohorts (future
  work) — raise the open-file limit instead.
- **`no_reference=True` is supported**, same as `from_vcf`/`from_pgen`: skips
  REF validation and left-alignment, reconstructing each atom's REF from the
  record's own REF bytes. The `reference`/`no_reference` exactly-one-of check
  and the `signatures`+`no_reference` incompatibility are otherwise identical
  to `from_vcf`.
  - **Caveat specific to this entry point:** because the merge is a per-contig
    k-way join keyed on each atom's normalized `(pos, ref, alt)`, skipping
    left-alignment means a site shared across files only joins into one output
    row if every input already represents it *identically* — same anchor base,
    same padding (e.g. all files came from the same caller, or were all
    already run through `bcftools norm` against the same reference). Two files
    encoding the same indel with different normalization will **not** join
    under `no_reference`: they silently become two separate variants in the
    output store instead of one shared row. This is not a failure mode that
    raises — verify upstream normalization is consistent before relying on
    `no_reference` with `from_vcf_list`.
- **`info_fields=`/`format_fields=`** — same declaration API as `from_vcf`
  (resolved against the FIRST file in `sources`'s header). Merge semantics
  differ from a single-file conversion because there are now N source
  columns per site:
  - **INFO fields merge first-carrier-wins.** When a site is shared across
    files, the stored INFO value comes from the lowest-numbered (earliest in
    `sources` order) file that carries the atom — not the last file, and not
    an aggregate (e.g. max/sum) of the carriers' values.
  - **FORMAT fields stay per-sample**, exactly as in `from_vcf`: each sample
    gets its own file's value; a sample that doesn't carry the atom at all
    gets the field's default (reserved sentinel/NaN, or an explicit
    `default=`).
- `ploidy`, `skip_out_of_scope`, `chunk_size`, `threads`, `overwrite`,
  `long_allele_capacity`, `signatures`, `check_ref` all mean the same as
  `from_vcf`, and the return value is the same `int` (dropped out-of-scope
  ALTs). `check_ref` is applied per input file during the merge (ignored
  when `no_reference=True`): under `"x"`, a bad record is excluded from its
  own file only (not the whole merged site), and the per-contig log reports
  the total excluded across every input file.

### Conversion from SVAR1

```python
from genoray import SparseVar2

dropped = SparseVar2.from_svar1(
    "out.svar2", "old.svar", "ref.fa",   # reference: validates REF + left-aligns indels
    overwrite=True,
)
```

Signature: `from_svar1(out, source, reference=None, *, regions=None, samples=None, merge_overlapping=False, regions_overlap="pos", no_reference=False, skip_out_of_scope=False, chunk_size=None, threads=None, overwrite=False, long_allele_capacity=8*1024*1024, signatures=False, check_ref="e") -> int`

Migrates an existing SVAR 1.0 (`SparseVar`) store to SVAR2 natively — reads no
VCF and no htslib; SVAR1 is already sparse, so this reconstructs variant
records from SVAR1's arrays and reuses the same conversion spine as `from_vcf`.

- `source` — a `SparseVar` store directory (SVAR1). `reference`/`no_reference`,
  `skip_out_of_scope`, `overwrite`, `long_allele_capacity`, `signatures`, and
  `check_ref` all mean the same as `from_vcf` (above), and return the same
  `int` (dropped out-of-scope ALTs).
- `ploidy` is read from SVAR1's metadata — no `ploidy=` kwarg.
- **Biallelic SVAR1 only** — raises `ValueError` if the source store has
  multiallelic variants (SVAR1's `geno==1` model); re-create the SVAR1 store
  biallelically first.
- **`regions=`/`merge_overlapping=`/`regions_overlap=`** — same convention,
  semantics, and three overlap modes (`"pos"`/`"record"`/`"variant"`) as
  `from_vcf`/`from_pgen`. `"variant"` mode keeps a record whole if ANY of its
  alleles truly overlaps the region (though SVAR1 is itself biallelic-only, so
  this only ever judges a single ALT). Unlike `from_pgen`, SVAR1 has no
  on-disk covering-range index to narrow against up front — a selected
  contig's local variants are still scanned in full; the per-record filter is
  what actually restricts the output, so this costs a full-contig scan rather
  than a range-restricted one.
- **`samples=`** — selects and reorders SVAR1 samples by name (same convention
  as `from_vcf`/`from_pgen`): preserves caller order, de-duplicates first
  occurrences, raises `ValueError` on an unknown name. `available_samples` and
  every decoded column match the caller's order exactly, regardless of each
  sample's original SVAR1 position.
- **Fields:** all SVAR1 FORMAT fields (e.g. `dosages`) carry through, keyed by
  their SVAR1 name. `mutcat` is dropped — pass `signatures=True` to recompute
  signatures from the reference instead of carrying SVAR1's.
- **Field parity caveat:** because SVAR1 never stored non-carrier FORMAT
  values, field output is byte-identical to `from_vcf` only for var_key
  (carrier-only) routed variants — for dense-routed variants, non-carrier
  cells are filled with the field's default/missing sentinel rather than the
  source VCF's true value. Genotype streams themselves (not fields) are
  byte-identical to `from_vcf` under matching normalization regardless of
  routing.
- **No `info_fields=`/`format_fields=` config kwargs** — unlike `from_vcf`,
  which fields to carry isn't configurable; every FORMAT field already present
  in the SVAR1 source is carried through automatically.

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
                                      # + one per selected field (see "Reading
                                      # INFO/FORMAT fields" below); shape
                                      # (R, S, P, None); pure-DEL ALT is empty

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

### Reading INFO/FORMAT fields (SVAR2)

Fields written by `from_vcf(info_fields=…, format_fields=…)` (above) are read
back by opting in — they are **not** decoded by default (each one costs extra
I/O).

```python
sv = SparseVar2("out.svar2")
sv.available_fields                    # {"AF": StoredField(...), "DS": StoredField(...)}

sv = sv.with_fields(["AF", "DS"])       # or SparseVar2("out.svar2", fields=["AF", "DS"])
rag = sv.decode("chr1", [(0, 10_000)])
rag["AF"]                              # Ragged, sharing offsets with pos/ilen/allele
```

- `SparseVar2(path, *, fields=None)` / `.with_fields(fields)` — `fields` is a
  `Sequence[str]` of canonical keys (see `available_fields` below).
  `with_fields` returns a **new** `SparseVar2` over the same store; it does
  not mutate the original in place. `fields=None` (the constructor default)
  selects nothing — fields are opt-in.
- `available_fields -> dict[str, StoredField]` — every field declared in the
  store's `meta.json`, keyed canonically: the bare field name when it is
  unique across INFO and FORMAT, else bcftools-style `INFO/DP` / `FORMAT/DP`
  when a name is used by both categories. `StoredField` (defined in
  `genoray._svar2_fields`, not exported at top-level `genoray`) is a frozen
  dataclass: `name`, `category` (`"info"`/`"format"`), `dtype` (`np.dtype`),
  `default` (`float | None`), `key`.
- `decode(contig, regions)` attaches one `Ragged` per selected field to the
  returned record `Ragged`, alongside `pos`/`ilen`/`allele` — every one
  sharing a single variant-axis offsets object, shape `(R, S, P, None)`.
  Access a field's data via `rag["KEY"]` (`Ragged.__getitem__`), **not**
  `rag.fields["KEY"]` — `Ragged.fields` is just the `list[str]` of field
  names on the record.
- **Dtype is preserved as stored.** SVAR2 losslessly auto-narrows integer
  fields at write time, so e.g. an `AC` field may come back as `int8`;
  nothing is widened on read.
- **Missing values** are the field's `default` if one was set at write time,
  else a reserved sentinel (`NaN` for floats, `iinfo.min`/`iinfo.max` for
  ints) — returned as-is, never translated.
- FORMAT fields are genotype-aligned (see `from_vcf`'s `format_fields=`
  above) — `decode()` only ever emits carrier records, so the "non-carrier
  values aren't stored" caveat from the write path is invisible on this
  read surface.

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

- `annotate_mutations(reference, *, gtf=None, contigs=None) -> None` —
  `reference` is a `genoray.Reference` or a FASTA path; `contigs=None`
  (default) annotates every contig. Unlike `SparseVar.annotate_mutations`,
  there is **no `write_back=` toggle** — SVAR2 always persists the sidecar to
  disk. `gtf=` optionally supplies a GTF/GFF gene model path; when given, each
  SNV is additionally classified by transcriptional-strand class (from
  `feature == "gene"` footprints) and persisted to a `strand.bin` sidecar,
  which unlocks the `"SBS192"`/`"SBS384"` catalogs below.
- `mutation_matrix(kind, *, count="allele"|"sample") -> pl.DataFrame` — a
  `MutationType` column (fixed COSMIC codebook order) plus one column per
  sample. `kind ∈ {"SBS96", "DBS78", "ID83", "SBS192", "SBS384"}`.
  `count="allele"` counts every non-ref allele copy; `count="sample"` counts
  each category at most once per sample, OR-combined across contigs. **Raises
  `ValueError`** if called before the store is annotated (no on-disk sidecar
  for every contig) — annotate first, either via `annotate_mutations` or
  `from_vcf(..., signatures=True)`. `"SBS192"`/`"SBS384"` additionally require
  strand annotation (`annotate_mutations(..., gtf=...)`) and raise
  `ValueError` if the store lacks it. `assign_signatures` does **not** accept
  `"SBS192"`/`"SBS384"` — see below.
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

### Strand-resolved catalogs (SBS192 / SBS384)

`SparseVar2` also supports the transcriptional-strand-bias catalogs, which
require a gene model (GTF) at annotation time:

```python
sv2.annotate_mutations(reference, gtf="gencode.v45.annotation.gtf.gz")
sbs384 = sv2.mutation_matrix("SBS384")   # 384 rows: [T, U, N, B] x 96
sbs192 = sv2.mutation_matrix("SBS192")   # 192 rows: the {T, U} sub-view = SBS384[:192]
```

- **SBS384** = 96 trinucleotide channels x 4 strand categories, SigProfiler
  order `[T, U, N, B]`: **T**ranscribed, **U**ntranscribed, **N**ontranscribed
  (intergenic), **B**idirectional (position covered by genes on both strands).
- **SBS192** is the `{T, U}` sub-view (`SBS384[:192]`).
- Strand rule (pyrimidine-folded): a genic SNV is Untranscribed iff the
  pyrimidine of its ref/alt pair sits on the gene's coding strand, else
  Transcribed. Gene footprints come from `feature == "gene"` rows (full gene
  body); pre-filter the GTF to restrict biotypes.
- Without a `gtf=`, `mutation_matrix("SBS192"/"SBS384")` raises. Write-time
  `from_vcf(..., signatures=True)` stays strand-free; obtain strand catalogs via
  a post-hoc `annotate_mutations(reference, gtf=...)`.
- `assign_signatures("SBS192"/"SBS384")` raises `NotImplementedError`: COSMIC
  publishes no strand-resolved reference set. Use `mutation_matrix` for
  strand-bias analysis.

### Merge and split by contig

SVAR2 contigs are fully independent on disk, so recombining or subsetting
whole contigs is a cheap metadata-rewrite + file-copy operation — unlike
`write_view` (see the CLI section below), none of these methods re-run
conversion or the var_key/dense cost model.

```python
from genoray import SparseVar2

sv = SparseVar2("out.svar2")
sv.subset_contigs("chr1.svar2", "chr1")                # single contig
sv.subset_contigs("subset.svar2", ["chr1", "chr2"])    # multiple, source order preserved
paths = sv.split_by_contig("by_contig/")               # one store per contig, out_dir/{contig}.svar2

SparseVar2.concat("merged.svar2", ["chr1.svar2", "chr2.svar2"])  # disjoint-contig merge
```

- `subset_contigs(output, contigs, *, mode="copy", overwrite=False) -> None` —
  write a new store containing only `contigs` (a single contig name or a
  sequence of names). Pure metadata rewrite + file copy of the kept contig
  directories, preserving the source store's contig order. Raises
  `ValueError` if any name isn't in `self.contigs`, or if `output` resolves
  to this store's own path (in-place subsetting is rejected, mirroring
  `write_view`'s in-place guard). Raises `FileExistsError` if `output`
  exists and `overwrite=False`.
- `split_by_contig(out_dir, *, mode="copy", overwrite=False) -> list[Path]` —
  explode into one single-contig store per contig at
  `out_dir/{contig}.svar2`; returns the output paths in `self.contigs`
  order. Implemented as one `subset_contigs` call per contig.
- `SparseVar2.concat(output, sources, *, mode="copy", overwrite=False) -> None`
  (classmethod) — concatenate stores with **disjoint** contig sets into one.
  `sources` is a sequence of paths (or `SparseVar2` instances); all sources
  must agree on `samples`, `ploidy`, `format_version`, and `fields` —
  disagreement on any of those, or a contig name appearing in more than one
  source, raises `ValueError`. The merged contig list is `natsorted`,
  independent of the order `sources` were passed in.
- `mode` (all three methods) is the shared `Mode` literal —
  `"copy"|"hardlink"|"symlink"|"move"` — controlling how each contig
  directory is transplanted into the output store.

### Errors

genoray raises standard Python builtins, by category:

- `ValueError` — bad input content: contig/sample not found, REF disagrees with
  the reference FASTA, or a symbolic/breakend ALT with `skip_out_of_scope=False`.
- `FileNotFoundError` — a required input file is missing.
- `OSError` — a corrupt/truncated store sidecar or an underlying disk I/O failure.
- `RuntimeError` — an internal genoray bug (a worker thread panicked); please
  report it.

## CLI

`genoray write` and `genoray view` each **default to SVAR2**; the previous
SVAR 1.0 behavior is available under the `svar1` subcommand of each
(`write svar1`, `view svar1`). `genoray concat`/`genoray split` are SVAR2-only
(no SVAR1 equivalent).

### `genoray write`

```bash
# SVAR2 (default) — reference required XOR --no-reference
genoray write file.vcf.gz out.svar2 --reference ref.fa
genoray write file.vcf.gz out.svar2 --no-reference
genoray write file.vcf.gz out.svar2 --reference ref.fa --skip-symbolics-and-breakends --threads 4
genoray write file.pgen out.svar2 --reference ref.fa
genoray write file.pgen out.svar2 --no-reference --regions chr1:1-1000 --samples A,B
genoray write store.svar out.svar2 --no-reference --samples A,B
genoray write vcf_dir/ out.svar2 --no-reference --regions chr1:1-1000

# SVAR 1.0 (previous default) — VCF or PGEN, dosages, --haploid, --max-mem
genoray write svar1 file.vcf.gz out.svar --max-mem 4g --haploid
```

- `genoray write` (SVAR2, thin wrapper over `SparseVar2.from_vcf`/`from_pgen`/
  `from_svar1`/`from_vcf_list`): resolves the source kind and dispatches —
  `.pgen` → `from_pgen`, a `.svar` directory → `from_svar1`, a directory (other
  than `.svar`) or any file that isn't a single `.vcf.gz`/`.bcf` → `from_vcf_list`
  (vcf-list form: a directory of single-sample VCFs/BCFs, or a manifest listing
  them), otherwise → `from_vcf`. `--regions`/`-r`, `--regions-file`/`-R`,
  `--merge-overlapping`, `--regions-overlap` (`pos`/`record`/`variant`) work
  for every source kind; `--samples`/`-s`, `--samples-file`/`-S` work for
  every source kind *except* the vcf-list form, which raises (each input file
  already contributes exactly one sample — there's no cohort to subset).
  `--reference` XOR `--no-reference` (required), `--chunk-size` (default 25000
  for VCF/BCF and vcf-list, memory-derived for PGEN and SVAR1), `--threads`/
  `-@`, `--overwrite`, `--long-allele-capacity` (advanced), and a single
  `--skip-symbolics-and-breakends` flag (maps to `skip_out_of_scope=`) — the
  SVAR2 core can't expand either symbolic ALTs (`<DEL>`, `<INS>`, …) or
  breakends into nucleotides, so they're dropped together; prints a `Dropped
  {n} out-of-scope (symbolic/breakend) ALT alleles.` line when set.
  `--ploidy` (default 2) applies to VCF/BCF and vcf-list sources only — PGEN
  is diploid and SVAR1's ploidy comes from its own metadata, so passing a
  non-default `--ploidy` with a `.pgen` source raises. `--check-ref {e,x}`
  (default `e`, maps to `check_ref=`): REF-vs-reference policy for every source
  kind, ignored with `--no-reference`. `e` aborts on the first REF/FASTA
  disagreement; `x` drops the offending record and continues. Mirrors
  `bcftools norm --check-ref`.
- `genoray write svar1`: unchanged SVAR 1.0 behavior — VCF or PGEN source,
  `--dosages`, `--max-mem`, `--haploid`, `--no-symbolic`/`--no-breakend`
  (independent flags here, unlike SVAR2).

### `genoray view`

```bash
# SVAR2 (default) — thin CLI over SparseVar2.write_view
genoray view in.svar2 out.svar2 -r chr1:1-1000 -s A,B
genoray view in.svar2 out.svar2 -r chr1:1-1000          # all samples
genoray view in.svar2 out.svar2 -s A,B                  # all variants (one region per contig)
genoray view in.svar2 out.svar2 -r chr1:1-1000 --no-reroute   # representation-preserving, low-memory view
genoray view in.svar2 out.svar2 -r chr1:1-1000 --reroute      # force the size-optimal re-route

# SVAR 1.0 (previous default) — unchanged SparseVar.write_view CLI
genoray view svar1 in.svar out.svar -r chr1:1-1000 -s A,B --progress
```

Both subcommands share the same `-r/--regions`, `-R/--regions-file`,
`-s/--samples`, `-S/--samples-file`, `-f/--fields`, `--merge-overlapping`,
`--regions-overlap`, `--overwrite`, `-@/--threads`, `--progress` options and
the same no-op guard (at least one of regions/samples is required) and mutex
checks (`--regions`/`--regions-file` and `--samples`/`--samples-file` are each
mutually exclusive).

- `genoray view` (SVAR2, thin wrapper over `SparseVar2.write_view`): when
  `--regions`/`--regions-file` is omitted, "all variants" defaults to one
  region per contig (`SparseVar2.contigs`, since SVAR2 has no contig-length
  metadata) spanning `[0, 2**31 - 1)` — every real POS is smaller. `--fields`
  defaults to `None`, meaning no fields are carried through (genotypes
  only) — this always succeeds, even on a store that has INFO/FORMAT fields.
  Both `--reroute` and `--no-reroute` go through the same slicer backend and
  carry `--fields`/`--reference` identically — there is no longer a
  fields-carrying vs. genotypes-only split between them:
  - `--reroute` reruns the var_key/dense routing cost model over the subset —
    *size-optimal* (each variant re-routed to whichever representation is
    smaller for the subset's sample/carrier counts).
  - `--no-reroute` (`reroute=False`) slices each variant's *existing*
    on-disk representation directly (no cost model, byte-level slice) —
    representation-preserving regardless of the subset's sample/carrier
    counts. Recommended for somatic/all-rare cohorts (nearly every variant
    is already var_key-routed) or memory-constrained runs.
  - Omitting both flags (the default) is `"auto"`: resolves to
    `--no-reroute`'s behavior when any FORMAT field is carried, to
    `--reroute`'s otherwise. WHY: a dense→var_key flip stores one value per
    *carrier call* and has no slot for a non-carrier sample's FORMAT value,
    so re-routing a source-dense variant under a FORMAT-carrying view would
    silently drop it — `"auto"` prefers fidelity whenever FORMAT is in play
    and takes the size-optimal re-route otherwise (genotype-only / INFO-only
    views have no per-sample slot to lose).

  Both `--reference` (recomputes `mutcat` from scratch on the subset) and
  `-@/--threads` (caps contigs sliced concurrently; autodetected when
  omitted) are real on both `--reroute` and `--no-reroute` — there is no
  longer an "accepted but ignored/unused" caveat on either path. `--progress`
  is accepted for parity but is currently a no-op on this path (see below).
  `write_view`'s underlying `reroute=` kwarg only accepts `"auto"`, `True`,
  or `False` — any other value (e.g. `reroute=1`) raises `ValueError` rather
  than silently falling through to the `reroute=False` slicer.
- `genoray view svar1`: unchanged SVAR 1.0 behavior — "all variants" defaults
  from `SparseVar`'s `_contig_stats` (`[0, pos_max + 1)` per contig); `--fields`
  defaults to all available fields (use an explicit empty selection to carry
  none); no `--reference`/`--reroute` options; `--progress` shows a real
  phase-level bar (see below).

### `genoray concat` / `genoray split`

```bash
genoray concat merged.svar2 part1.svar2 part2.svar2       # disjoint-contig merge
genoray split in.svar2 out_dir/                           # explode into out_dir/{contig}.svar2
genoray split in.svar2 subset.svar2 --contigs chr1,chr2    # subset into one store
```

Both accept `--mode` (`Literal["copy", "hardlink", "symlink", "move"]`, default
`"copy"` — see `SparseVar2.concat`/`split_by_contig`/`subset_contigs`
docstrings) and `--overwrite`.

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
unaffected. The `genoray view svar1` CLI exposes the same option as `--progress`
(also default off):

```bash
genoray view svar1 in.svar out.svar -r chr1:1-1000 -s A,B --progress
```

The bar is cosmetic: output bytes, schema, and dtypes are identical whether or
not it is enabled.

`SparseVar2.write_view` also accepts `progress=`/`--progress` for interface
parity, but it is currently a no-op there (no bar shown) — see the `genoray
view` CLI section above.

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
  or call `annotate_mutations` on the output view yourself. (This is
  `SparseVar`/v1 behavior; on `SparseVar2.write_view`, `reference=` recomputes
  `mutcat` from scratch on the subset on **both** `reroute=True` and
  `reroute=False`.)
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

- No strand-bias separation (SBS-192 / SBS-384) — v1 `SparseVar` only. Use
  `SparseVar2.annotate_mutations(reference, gtf=...)` for transcriptional
  strand-resolved catalogs.
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
| Calling `write_view` and expecting `mutcat` to be in the output | `write_view` **never** copies `mutcat` positionally (subsetting invalidates DBS adjacency codes). Pass `reference=` to `write_view` to recompute it on the subset, or call `annotate_mutations` on the output view yourself. Explicitly including `"mutcat"` in `fields=` without a `reference=` raises `ValueError`. On `SparseVar2.write_view`, `reference=` recomputes `mutcat` from scratch on both `reroute=True` and `reroute=False`. |
| Passing a FORMAT field in `fields=` to `SparseVar2.write_view(..., reroute=True)` and expecting it dropped/rejected | Both `reroute=True` and `reroute=False` carry `fields` through now (previously `reroute=True` raised `ValueError`). Watch the `reroute="auto"` default instead: it resolves to `reroute=False` whenever any FORMAT field is carried, because a dense→var_key flip has no slot for a non-carrier sample's FORMAT value. |
| Passing the source dataset directory as `output` to `write_view` (even with `overwrite=True`) | Raises `ValueError` — writing in place would delete the source before the view is written. Pass a different `output` path. |
| Passing a FASTA path directly to `annotate_mutations` | Supported — it auto-wraps via `Reference.from_path` |
| `rag.fields["AF"]` on a `SparseVar2.decode()` result | `Ragged.fields` is a `list[str]` of names, not a mapping; index the field itself with `rag["AF"]` |
| Expecting `SparseVar2.decode()` to include INFO/FORMAT fields | Fields are opt-in — pass `fields=[...]` to `SparseVar2(...)` or call `.with_fields([...])` first |

## When this skill needs updating

Any PR that adds, removes, renames, or changes the semantics of a public
name (anything reachable from `import genoray` without underscores) must
update this skill alongside the code change. See the project `CLAUDE.md`.

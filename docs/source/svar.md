# Sparse Variant Format (SVAR)

## Motivation

Typical genomic data formats such as VCF/BCF and PLINK encode genotypes in a dense matrix. However, these matrices are typically extremely sparse (< 1% density), especially with whole genome sequencing or cancer data. To avoid consuming excessive amounts of disk space, these formats use block-wise compression. However, block compressed data can easily create data processing bottlenecks in machine learning applications, where random sampling is required to during training. This was a huge problem during the development of [GenVarLoader](https://github.com/mcvickerlab/GenVarLoader), for example. By instead using a sparse format, we were able to circumvent compression while keeping the file size on par with an equivalent compressed BCF. As a result, we can memory map the genotypes to work with larger-than-RAM data with random access that is much, much faster than compressed formats. For example, GenVarLoader computes direclty on the SVAR format and this is a major factor in its 1000x speedup over alternative methods.

## Creating SVAR files

```python
from genoray import SparseVar, VCF, PGEN

SparseVar.from_vcf("out.svar", "file.vcf.gz", max_mem="4g")
SparseVar.from_pgen("out.svar", "file.pgen", max_mem="4g")

svar = SparseVar("out.svar")
```

### Region/sample-restricted SVAR2 conversion

`SparseVar2.from_vcf`, `from_pgen`, and `from_svar1` can all convert directly
into a subset of regions and samples. `from_vcf_list` supports the same
`regions=`/`merge_overlapping=`/`regions_overlap=` but has **no `samples=`**
— each input file is single-sample, so the cohort is defined by the file set
itself:

```python
from genoray import SparseVar2

SparseVar2.from_vcf(
    "subset.svar2",
    "cohort.vcf.gz",
    "reference.fa",
    regions=["chr1:1-1000000", ("chr2", 0, 500_000)],
    samples=["HG00096", "HG00097"],
    merge_overlapping=True,
    threads=8,
)
```

Region strings use bcftools-style 1-based inclusive coordinates and are
converted to 0-based half-open intervals. Tuple, BED, and frame inputs are
already interpreted as 0-based half-open. `samples` selects and reorders by
name, preserving caller order and deduplicating repeated names by first
occurrence.

The equivalent CLI flags are available on the default SVAR2 writer, for every
source kind (VCF/BCF, PGEN, an SVAR1 store, or a vcf-list directory/manifest):

```bash
genoray write cohort.vcf.gz subset.svar2 \
  --reference reference.fa \
  --regions chr1:1-1000000,chr2:1-500000 \
  --samples HG00096,HG00097 \
  --threads 8
```

Use `--regions-file/-R` for BED files and `--samples-file/-S` for
one-sample-per-line sample lists. `--samples`/`-s`/`--samples-file`/`-S` are
rejected for the multi-file (vcf-list) directory/manifest form — each input
file already contributes exactly one sample, so there's no cohort left to
subset.

`regions_overlap` selects one of three overlap modes, matching bcftools
`--regions-overlap`: `"pos"` (default; POS inside `[start,end)`), `"record"`
(POS in `[start,end+1)`, so an indel at the region's last base is kept), or
`"variant"` (the anchor-trimmed variant extent overlaps the region). In
`variant` mode a multiallelic record is kept whole if ANY of its alleles
truly overlaps the region; individual non-overlapping alleles are not
dropped.

## Haploid (ploidy=1) write option

For unphased somatic cohorts where phasing information is unavailable or irrelevant,
`from_vcf` and `from_pgen` accept `haploid=True` to OR-collapse both haplotypes into a
single haploid call per sample. A variant is recorded for a sample if it is present on
*either* haplotype. The resulting SVAR stores `ploidy=1` in its metadata, and all
downstream read operations (including `write_view` and `annotate_mutations`) work
transparently at ploidy=1.

```python
from genoray import SparseVar, VCF

# Collapse diploid genotypes to haploid (ploidy=1 in output)
SparseVar.from_vcf("out.svar", VCF("file.vcf.gz"), max_mem="4g", haploid=True)

svar = SparseVar("out.svar")
assert svar.ploidy == 1
# shape: (ranges, samples, ploidy=1, ~variants)
rag = svar.read_ranges("chr1", starts=[0], ends=[1_000_000])
```

The equivalent CLI command:

```bash
genoray write file.vcf.gz out.svar --max-mem 4g --haploid
```

## Reading SVAR

```python
# shape: (ranges, samples, ploidy, ~variants)
sp_genos = svar.read_ranges("1", starts=0, ends=365, samples="Aang")
```

When using an SVAR file, `read_ranges` returns a `Ragged[V_IDX_TYPE]` — a ragged array where
the number of ALT calls per sample and ploid varies. For a brief visual description of Ragged
arrays, see [this section of the GenVarLoader FAQ](https://genvarloader.readthedocs.io/en/latest/faq.html#why-does-a-dataset-return-ragged-objects-and-what-are-they).
The returned array can be arbitrarily large because its data is backed by a
[`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) object
(only the offsets reside in RAM).

Each value in the ragged array is a variant index: the row number in `svar.index` for the
variant that is present in each range, sample, and ploid.

```python
v_idxs = sp_genos.to_awkward()[0, 0, 0].to_numpy()
```

## Loading additional fields

Custom numeric fields stored as `.npy` files in the SVAR directory can be loaded alongside
genotype indices. Only VCF FORMAT fields with `Number=G` are currently supported.

```python
# Load at construction time
svar = SparseVar("out.svar", fields={"dosages": np.float32})

# Or derive from an existing SparseVar (shallow copy, re-opens the memmaps)
svar_with = svar.with_fields({"dosages": np.float32})

# read_ranges now returns an awkward record array
result = svar_with.read_ranges("1", starts=0, ends=365)
result.genos.data    # flat array of variant indices (uint32)
result.dosages.data  # flat array of dosage values (float32)

# Drop all fields to get back a plain Ragged[V_IDX_TYPE]
svar_plain = svar_with.with_fields(False)
```

There's a lot more that can be done with `SparseVar`; this documentation will be expanded as time permits.

## Mutational signatures (SBS-96 / DBS-78 / ID-83)

`SparseVar` and its next-gen successor `SparseVar2` both support COSMIC-style
mutation catalogue annotation and signature refitting via
`annotate_mutations`, `mutation_matrix`, and `assign_signatures`
(`SparseVar2` can also classify during conversion with
`SparseVar2.from_vcf(..., signatures=True)`). This isn't narrated here yet —
see the `genoray-api` skill (`skills/genoray-api/SKILL.md`, "Mutation
catalogues" and "SparseVar2 — quick reference → Mutational signatures"
sections) or the method docstrings in `genoray/_svar/_annotate.py` (v1) and
`genoray/_svar2_mutcat.py` (SVAR2) for the full workflow.

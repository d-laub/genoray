# `genoray`

If you want to use NumPy with genetic variant data, `genoray` is for you! `genoray` provides a uniform API for efficient range queries of genotypes and dosages from VCF and PGEN (PLINK 2.0) files. `genoray` is also fully type-safe and has minimal dependencies.

The API is minimal, with only three core methods for single range queries, automatically chunking reads **to respect a memory limit**, and reading **multiple range queries**.

## `read`
Read genotypes and/or dosages for a single range `(contig, start, end)`. Returns a NumPy array of shape `(samples, ploidy, variants)`. For VCF, the dtype is `np.int8`, and for PGEN, the dtype is `np.int32`.

```python
from genoray import VCF, PGEN

vcf = VCF("file.vcf.gz")
pgen = PGEN("file.pgen")

# shape: (samples ploidy variants)
genos = vcf.read("1")  # read all variants on chromosome 1
genos = pgen.read("1")
```

You can also change the return type to be either genotypes and/or dosages using the read_as argument in the constructor:

```python
from genoray import VCF, PGEN

vcf = VCF("file.vcf.gz", read_as=VCF.GenosDosages)  # can be VCF.Genos, VCF.Dosages, or VCF.GenosDosages
pgen = PGEN("file.pgen", read_as=PGEN.GenosDosages)

# shape: (samples ploidy variants), shape: (samples variants)
genos, dosages = vcf.read("1")
genos, dosages = pgen.read("1")
```

Dosages have shape `(samples, variants)` and dtype `np.float32` for both VCF and PGEN.

> [!IMPORTANT]
> PGEN files are automatically indexed on construction, creating a `<prefix>.gvi` file. This is a one-time cost to enable much faster range queries, but it takes longer for larger files. Don't delete this index file unless you want to re-index the PGEN file.

## `read_chunks`
If you don't have enough memory to read a range of variants, you can split the read into chunks. This method returns a generator that yields NumPy arrays of shape `(samples, ploidy, variants)` (and/or `(samples, variants)` for dosages) that are automatically chunked along the `variants` axis such that each chunk of data is no larger than the specified maximum memory.

```python
genos = vcf.read_chunks("1", max_mem="4g")  # default is "4g", can also be capitalized or be "GB", for example
```

## `read_ranges`
If you want to read multiple ranges on the same contig at once, you can use the `read_ranges` method. This method takes starts and ends and returns a NumPy array of shape `(samples, ploidy, variants)` (and/or `(samples, variants)` for dosages) for each range, as well as the number of variants in each range. Since the data is allocated as a single array, the number of variants in each range let's you slice out the data for each range from the `variants` axis.

```python
# shape: (samples, ploidy, variants), shape: (n_ranges)
genos, n_vars = vcf.read_ranges('1', starts=[1, 1000, 2000], ends=[1000, 2000, 3000])
```

## Filtering

You can filter variants from VCF or PGEN files by a providing a function or [polars expression](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) to the constructor, respectively. For VCFs, the function must accept a [cyvcf2.Variant](https://brentp.github.io/cyvcf2/docstrings.html#cyvcf2.cyvcf2.Variant) and return a boolean indicating whether to include the variant. For PGENs, the expression will operate on a polars DataFrame with four columns: "Chromosome", "Start", "End", and "kind". The expression should return a boolean mask indicating which variants to include. The "kind" column is a string that indicates the type of variant, which can be "SNP", "INDEL", or "MNP".

```python
vcf = VCF("file.vcf.gz", filter=lambda v: v.QUAL > 20)  # only include variants with quality > 20
pgen = PGEN("file.pgen", filter=pl.col("kind") == "SNP")  # only include SNPs
```

# ⚠️ Important ⚠️

- For the time being, ploidy is always 2, but this could be more flexible for VCFs in the future. PGEN does not support ploidy other than 2.
- Multi-allelic variants are not supported.
- PGEN returns genotypes with dtype `np.int32` instead of `np.int8` because this is the native dtype for pgenlib.
- Ranges are 0-based, so starts begin at 0 and ends are exclusive.
- Missing genotypes and dosages are encoded as -1 and `np.nan`, respectively.
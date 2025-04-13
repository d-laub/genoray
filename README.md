# `genoray`

If you want to use NumPy with genetic variant data, `genoray` is for you! `genoray` provides a uniform API for efficient range queries of genotypes and dosages from VCF and PGEN (PLINK 2.0) files. `genoray` is also fully type-safe and has minimal dependencies.

The API is minimal, with only three core methods: **single range queries**, automatically chunking a range query **to respect a memory limit**, and reading **multiple range queries**.

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

Note that VCFs must also be provided a FORMAT `dosage_field` to read dosages and this field must have `Number=A` in the header, meaning there is one value for each ALT allele. PGEN files either store hardcalls (genotypes) or dosages, not both, and dosage PGENs infer hardcalls based on a [hardcall threshold](https://www.cog-genomics.org/plink/2.0/input#dosage_import_settings). Thus, if you want to read hardcalls that do not correspond to inferred hardcalls from a dosage PGEN, you can provide two different PGEN files to the constructor:

```python
pgen = PGEN("hardcalls.pgen", dosage_path="dosage.pgen", ...)
```

This will read hardcalls from `hardcalls.pgen` and dosages from `dosage.pgen`. The two PGEN files must have the same samples and variants in the same order. The `dosage_path` argument is optional, and if not provided, both hardcalls and dosages will be sourced from the path argument (`"hardcalls.pgen"` in the example).

> [!IMPORTANT]
> PGEN files are automatically indexed on construction, creating a `<prefix>.gvi` file. This is a one-time cost to enable much faster range queries, but it takes longer for larger files. Don't delete this index file unless you want to re-index the PGEN file.

## `read_chunks`
If you don't have enough memory to read a range of variants, you can split the read into chunks. This method returns a generator that yields NumPy arrays of shape `(samples, ploidy, variants)` (and/or `(samples, variants)` for dosages) that are automatically chunked along the `variants` axis such that each chunk of data is no larger than the specified maximum memory.

```python
genos = vcf.read_chunks("1", max_mem="4g")  # default is "4g", can also be capitalized or be "GB", for example
```

## `read_ranges`
If you want to read multiple ranges on the same contig at once, you can use the `read_ranges` method. This method takes starts and ends and returns a NumPy array of shape `(samples, ploidy, variants)` (and/or `(samples, variants)` for dosages) for each range, as well as the offsets to slice out the variants for each range. Since the data is allocated as a single array, the offsets let you slice out the data for each range from the `variants` axis.

```python
# shape: (samples, ploidy, variants), shape: (n_ranges+1)
genos, offsets = vcf.read_ranges('1', starts=[1, 1000, 2000], ends=[1000, 2000, 3000])
first_range_genos = genos[..., offsets[0]:offsets[1]]
```

## Filtering

You can filter variants from VCF or PGEN files by a providing a function or [polars expression](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) to the constructor, respectively. For VCFs, the function must accept a [cyvcf2.Variant](https://brentp.github.io/cyvcf2/docstrings.html#cyvcf2.cyvcf2.Variant) and return a boolean indicating whether to include the variant. For PGENs, the expression will operate on a polars DataFrame with four columns: "Chromosome", "Start", "End", and "kind". The expression should return a boolean mask indicating which variants to include. The "kind" column is a string that indicates the type of variant, which can be "SNP", "INDEL", or "MNP".

```python
vcf = VCF("file.vcf.gz", filter=lambda v: v.QUAL > 20)  # only include variants with quality > 20
pgen = PGEN("file.pgen", filter=pl.col("kind") == "SNP")  # only include SNPs
```

## Type Safety

`genoray` is fully type-safe, meaning that type checkers like mypy and pyright can infer the types of the input and output data for everything in `genoray` and provide compile-time errors. In addition, if you want to abstract your code to work with any `genoray` reader, you can use the `genoray.Reader` base class, which is the `typing.Protocol` that all `genoray` readers adhere to. This allows you to write code that works with either VCF or PGEN files without having to worry about the underlying implementation details.

# ⚠️ Important ⚠️

- For the time being, ploidy is always 2, but this could be more flexible for VCFs in the future. PGEN does not support ploidy other than 2.
- Multi-allelic variants are not supported and any multi-allelic sites will only have their first allele returned. As a workaround, you can split multi-allelic sites into bi-allelic ones using `bcftools norm` or `plink2 --make-bpgen`. You can then pass the PLINK 1 `.bed` file to `genoray.PGEN`, although this will erase phasing information. If you need the phasing information, you should export the file to BCF and split the sites with `bcftools`, then convert the BCF back to PGEN with `plink2 --make-pgen`.
- PGEN returns genotypes with dtype `np.int32` instead of `np.int8` because this is the native dtype for pgenlib.
- Ranges are 0-based, so starts begin at 0 and ends are exclusive.
- Missing genotypes and dosages are encoded as -1 and `np.nan`, respectively.

# Contributing

To contribute to `genoray`, please fork the repository and create a pull request. We welcome contributions of all kinds, including bug fixes, new features, and documentation improvements. Please make sure to run the tests before submitting a pull request. We provide a Pixi environment that includes all development dependencies. To use the environment, install Pixi and run `pixi run pre-commit` to activate pre-commit in your clone of the repo, and then run `pixi s` in the repository root directory. `pixi s` will activate the development environment and install all dependencies. You can then run the tests using `pytest`. ❗Note that all commits must adhere to [conventional commits](https://www.conventionalcommits.org/). If you have any questions or suggestions, please open an issue on the repository.
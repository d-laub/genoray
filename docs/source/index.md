```{toctree}
:hidden: true

svar
api
```

# `genoray`

If you want to use NumPy with genetic variant data, `genoray` is for you! `genoray` enables ergonomic and efficient range queries of genotypes and dosages from VCF and PGEN (PLINK 2.0) files. `genoray` is also fully type-safe and has minimal dependencies.

# Summary

The `genoray` API more-or-less boils down to just two classes and up to four methods:

- `VCF` and `PGEN` classes for reading VCF and PGEN files, respectively.
- `read` method for reading a single range of variants.
- `chunk` method for reading a single range of variants in chunks.
- `read_ranges` method for reading multiple ranges of variants at once.
- `chunk_ranges` method for reading multiple ranges of variants at once in chunks.

The other important arguments to know are `mode` (and `phasing` for VCF) to set the return type and `max_mem` for chunking. The modes that are available for each file format are always accessible from the class itself, e.g. `VCF.Genos16`, `PGEN.GenosDosages`, etc. Lastly, you can filter variants on the fly using the `filter` argument to class constructors.

# Examples

## VCF

We work with VCFs using the (you guessed it) VCF class:

```python
from genoray import VCF

vcf = VCF("file.vcf.gz")
```

Querying data for a region is as simple as:

```python
# shape: (samples ploidy variants)
genos = vcf.read("1")  # read all variants on chromosome 1
```

You can also change the return type to be either genotypes and/or dosages by providing a `mode` argument:

```python
vcf = VCF("file.vcf.gz", dosage_field="DS")  # need a dosage_field to read dosages

genos, dosages = vcf.read("1", mode=VCF.Genos16Dosages)
```

Dosages have shape `(samples, variants)` and dtype `np.float32`.

```{note}
VCFs must also be provided a FORMAT `dosage_field` to read dosages and this field must have `Number=A` in the header, meaning there is one value for each ALT allele. 
```

A key feature of `genoray` is letting you work with data that is too large to fit into memory. For example:

```python
vcf = VCF("file.vcf.gz", phasing=True)  # include phasing status

# max_mem defaults to "4g", can also be capitalized or be "GB", for example
# Genos8 reduces precision to int8 from the default int16 that cyvcf2 uses
genos = vcf.chunk("1", max_mem="4g", mode=VCF.Genos8)

for chunk in genos:
    # do something with chunk, each chunk is a NumPy array of shape (samples, ploidy+1, variants)
    ...
```
The `chunk` method will automatically chunk the data along the `variants` axis to respect the memory limit, returning a generator of data instead of everything at once.

```{note}
We also set `phasing=True` and changed the mode to `VCF.Genos8` to read phased genotypes as `int8`. The `phasing` argument lets us have access to the phase of the genotype in the format the cyvcf2 adheres to: the 3rd entry along the ploidy axis is the phase: 0 for unphased, 1 for phased. Reducing precision to `int8` instead of `int16` reduces the memory per variant by half -- we would only need higher precision if we expected to have more than 128 alleles at a variant site.
```

## PGEN

```python
from genoray import PGEN

pgen = PGEN("file.pgen")
```

```{important}
PGEN files are automatically indexed on construction, creating a `<prefix>.gvi` file. This is a one-time cost to enable fast range queries, but it takes longer for larger files. Don't delete this index file unless you want to re-index the PGEN file.
```

We can query data for a region in the same way as VCF:

```python
# shape: (samples ploidy variants)
genos = pgen.read("1")  # read all variants on chromosome 1
genos = pgen.chunk("1")  # read all variants on chromosome 1
```

However, PGEN files also support reading multiple ranges at once since this improves throughput substantially:

```python
# shape: (samples, ploidy, variants), shape: (n_ranges+1)
genos, offsets = pgen.read_ranges('1', starts=[1, 1000, 2000], ends=[1000, 2000, 3000])
first_range_genos = genos[..., offsets[0]:offsets[1]]

genos = pgen.chunk_ranges('1', starts=[1, 1000, 2000], ends=[1000, 2000, 3000])
for range_ in genos:
    if range_ is None:
        # no data for this range
        continue
    for chunk in range_:
        # do something with chunk, each chunk is a NumPy array of shape (samples, ploidy, variants)
        ...
```

The `read_ranges` method takes starts and ends and returns data for each range and the offsets to slice out the variants for each range. Since the data is allocated as a single array, the offsets let you slice out the data for each range from the `variants` axis.

```{note}
We do not provide an API for multi-range queries of VCFs because benchmarking showed that this provided no benefit to throughput.
```

Like VCF, methods for PGENs accept a `mode` argument to change the return type to include genotypes, phasing, and/or dosages:

```python
genos, phasing, dosages = pgen.read("1", mode=PGEN.GenosPhasingDosages)
```

The PGEN reader adheres to pgenlib's API, so the phasing information is in a separate boolean array instead of using an extra column like VCF/cyvcf2. The phasing information is a boolean array of shape `(samples, variants)` where `True` indicates that the genotype is phased and `False` indicates that it is unphased.

:::{important}
PGEN files either store hardcalls (genotypes) or dosages, not both, and dosage PGENs infer hardcalls based on a [hardcall threshold](https://www.cog-genomics.org/plink/2.0/input#dosage_import_settings). Thus, if you want to read hardcalls that do not correspond to inferred hardcalls from a dosage PGEN, you can provide two different PGEN files to the constructor:

```python
pgen = PGEN("hardcalls.pgen", dosage_path="dosage.pgen", ...)
```

This will read hardcalls from `hardcalls.pgen` and dosages from `dosage.pgen`. The two PGEN files must have the same samples and variants in the same order. The `dosage_path` argument is optional, and if not provided, both hardcalls and dosages will be sourced from the path argument (`"hardcalls.pgen"` in the example).
:::

## Filtering

You can filter variants from VCF or PGEN files by a providing a function or [polars expression](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/) to the constructor, respectively.

For VCFs, the function must accept a [cyvcf2.Variant](https://brentp.github.io/cyvcf2/docstrings.html#cyvcf2.cyvcf2.Variant) and return a boolean indicating whether to include the site.

```python
# only include variants that are common in EUR
vcf = VCF("file.vcf.gz", filter=lambda v: v.INFO['AF_EUR'] > 0.05)
```

For PGENs, the expression will operate on a polars DataFrame with all of the columns available in the underlying PVAR except `#CHROM`, `POS`, and `ALT`, which are superseded by columns added by `genoray`:
- `Chromosome`
- `Start`
- `End`
- `ALT` as a list of strings
- `ilen` (indel length)
- `kind` a list of strings that indicates the type of each ALT allele as `"SNP"`, `"INDEL"`, `"MNP"`, or `"OTHER"`

The expression should return a boolean mask indicating which variants to include.

```python
# only include SNPs
pgen = PGEN("file.pgen", filter=pl.col("kind").list.eval(pl.element() == "SNP").list.all())
```

# ⚠️ Important ⚠️

- For the time being, ploidy is 2 for all classes in `genoray`, but this could be more flexible for VCFs in the future. The PGEN format does not support ploidy other than 2.
- Different file formats may use different data types for their respective representations of genotypes, phasing, and dosages.
- Ranges are 0-based, so starts begin at 0 and ends are exclusive.
- Missing genotypes and dosages are encoded as -1 and `np.nan`, respectively.
- Dosages from PGEN files may not exactly match VCF files (up to a fraction of a percent) because PLINK 2.0 must encode dosages with fixed precision which can not match what can be represented by text in a VCF (may also disagree with how BCF encodes dosage).

# Contributing

To contribute to `genoray`, please fork the repository and create a pull request. We welcome contributions of all kinds, including bug fixes, new features, and documentation improvements. Please make sure to run the tests before submitting a pull request. We provide a Pixi environment that includes all development dependencies. To use the environment, install Pixi and run `pixi run pre-commit` to activate pre-commit in your clone of the repo, and then run `pixi s` in the repository root directory. `pixi s` will activate the development environment and install all dependencies. You can then run the tests using `pytest`. ❗Note that all commits must adhere to [conventional commits](https://www.conventionalcommits.org/). If you have any questions or suggestions, please open an issue on the repository.
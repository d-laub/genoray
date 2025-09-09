"""`Polars <https://docs.pola.rs/>`_ expressions for filtering a genoray index (extension :code:`.gvi`)
given the minimum set of index columns:

- :code:`"CHROM"` : :code:`pl.Utf8`
- :code:`"POS"` : :code:`pl.Int64`
- :code:`"REF"` : :code:`pl.Utf8`
- :code:`"ALT"` : :code:`pl.List[Utf8]`
- :code:`"ILEN"` : :code:`pl.List[Int32]`

Applicable for PGEN files and the experimental :meth:`VCF._load_index` method.

.. note::
    For PGEN, all columns that existed in the underlying PVAR will be available in the index.
"""

import polars as pl

IndexSchema = {
    "CHROM": pl.Categorical,
    "POS": pl.Int64,
    "REF": pl.Utf8,
    "ALT": pl.List(pl.Utf8),
    "ILEN": pl.List(pl.Int32),
}
"""Minimum schema for a genoray index file (extension :code:`.gvi`)."""

is_snp = pl.col("ILEN").list.eval(pl.element() == 0).list.all()
"""True if all ALT alleles are SNPs (single nucleotide polymorphisms)."""

is_indel = pl.col("ILEN").list.eval(pl.element() != 0).list.all()
"""True if all ALT alleles are indels (insertions or deletions)."""

is_biallelic = pl.col("ALT").list.len() == 1
"""True if the variant is biallelic (one ALT allele)."""

ILEN = pl.col("ALT").list.eval(pl.element().str.len_bytes().cast(pl.Int32)) - pl.col(
    "REF"
).str.len_bytes().cast(pl.Int32)
"""Indel length of the variant. Positive for insertions, negative for deletions, and zero for SNPs and MNPs."""

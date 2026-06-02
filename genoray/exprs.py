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

"""
on-disk schema is same as in-memory except:
CHROM is cat or enum
ALT can be comma delimited str or list[str]
ILEN is optional list[int]
"""

# in-memory schema
IndexSchema = {
    "CHROM": pl.Enum,
    "POS": pl.Int64,
    "REF": pl.Utf8,
    "ALT": pl.List(pl.Utf8),
    "ILEN": pl.List(pl.Int32),
}
"""Minimum in-memory schema for a genoray index file (extension :code:`.gvi`)."""

is_snp = pl.col("ILEN").list.eval(pl.element() == 0).list.all()
"""True if all ALT alleles are SNPs (single nucleotide polymorphisms)."""

is_indel = pl.col("ILEN").list.eval(pl.element() != 0).list.all()
"""True if all ALT alleles are indels (insertions or deletions)."""

is_biallelic = pl.col("ALT").list.len() == 1
"""True if the variant is biallelic (one ALT allele)."""

is_symbolic = pl.col("ALT").list.eval(pl.element().str.starts_with("<")).list.any()
"""True if any ALT allele is a symbolic allele (e.g. :code:`<DEL>`, :code:`<INS>`,
:code:`<DUP>`, :code:`<INV>`, :code:`<CNV>`, :code:`<BND>` — anything matching ``<…>``
per the VCF 4.x spec).

Symbolic ALTs are placeholders for structural variants whose exact replacement
nucleotides are unknown. Downstream haplotype injection (e.g. via
``genvarloader``) cannot expand them — the literal ``<DEL>`` ASCII bytes end up
in personalized DNA buffers and become non-canonical bytes for translators.
Pair with :class:`~genoray.VCF`'s ``skip_symbolic_alts`` constructor flag to
filter these out, or compose directly with other expressions.

Example
-------
>>> import genoray
>>> vcf = genoray.VCF("file.vcf.gz", pl_filter=~genoray.exprs.is_symbolic,
...                   filter=lambda v: not v.ALT[0].startswith("<"))  # doctest: +SKIP
"""

ILEN = pl.col("ALT").list.eval(pl.element().str.len_bytes().cast(pl.Int32)) - pl.col(
    "REF"
).str.len_bytes().cast(pl.Int32)
"""Indel length of the variant. Positive for insertions, negative for deletions, and zero for SNPs and MNPs."""

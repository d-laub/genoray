"""`Polars <https://docs.pola.rs/>`_ expressions for filtering a genoray index (extension :code:`.gvi`).

These require the minimum set of index columns:

- :code:`"CHROM"` : :code:`pl.Utf8`
- :code:`"POS"` : :code:`pl.Int64`
- :code:`"REF"` : :code:`pl.Utf8`
- :code:`"ALT"` : :code:`pl.List[Utf8]`
- :code:`"ILEN"` : :code:`pl.List[Int32]`

Applicable to PGEN indexes, and to VCF indexes when one has been built.

.. note::
    For PGEN, all columns that existed in the underlying PVAR will be available in the index.
"""

import re
from collections.abc import Iterable

import polars as pl

"""
on-disk schema is same as in-memory except:
CHROM is cat or enum
ALT can be comma delimited str or list[str]
ILEN is optional list[int]
"""

# in-memory schema
_IndexSchema = {
    "CHROM": pl.Enum,
    "POS": pl.Int64,
    "REF": pl.Utf8,
    "ALT": pl.List(pl.Utf8),
    "ILEN": pl.List(pl.Int32),
}
"""Minimum in-memory schema for a genoray index file (extension :code:`.gvi`)."""

is_snp = (
    pl.col("ILEN")
    .list.eval((pl.element() == 0) & pl.element().is_not_null())
    .list.all()
)
"""True if all ALT alleles are SNPs (single nucleotide polymorphisms).

Un-sizable symbolic alleles (``null`` ILEN) are treated as neither SNP nor
indel: a row containing any ``null`` ILEN element evaluates to ``False``.
"""

is_indel = (
    pl.col("ILEN")
    .list.eval((pl.element() != 0) & pl.element().is_not_null())
    .list.all()
)
"""True if all ALT alleles are indels (insertions or deletions).

Un-sizable symbolic alleles (``null`` ILEN) are treated as neither SNP nor
indel: a row containing any ``null`` ILEN element evaluates to ``False``.
"""

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

To drop symbolic records, pass this as a filter. For PGEN, the single ``filter``
expression suffices::

    pgen = genoray.PGEN("file.pgen", filter=~genoray.exprs.is_symbolic)

For VCF, bundle it with the equivalent cyvcf2 record predicate in a
:class:`genoray.Filter` (both are required)::

    vcf = genoray.VCF(
        "file.vcf.gz",
        filter=genoray.Filter(
            record=lambda rec: not any(a.startswith("<") for a in rec.ALT),
            expr=~genoray.exprs.is_symbolic,
        ),
    )

``SparseVar.from_vcf`` / ``from_pgen`` inherit the source's filter, so the SVAR
is filtered to match.
"""

_BND_PATTERN = r"[\[\]]|^\.[A-Za-z]|[A-Za-z]\.$"
"""Regex matching a VCF breakend (BND) ALT replacement string.

Catches the paired mate forms (``t[p[``, ``t]p]``, ``]p]t``, ``[p[t`` — all
contain ``[`` or ``]``) and the single-breakend forms (``.t`` / ``t.`` — a ``.``
adjacent to a base). A lone ``.`` (no-ALT) does not match because a base must
sit beside the dot."""

is_breakend = (
    pl.col("ALT").list.eval(pl.element().str.contains(_BND_PATTERN)).list.any()
)
"""True if any ALT allele is a breakend (BND) in mate-pair / single-breakend
notation (e.g. :code:`G[chr2:321[`, :code:`]chr2:321]G`, :code:`.TGCA`,
:code:`TGCA.`), per the VCF 4.x spec (§5.4).

Breakends are a *distinct* ALT class from symbolic ``<...>`` alleles, so
:data:`is_symbolic` does **not** flag them. But like symbolic alleles they are
not expandable into nucleotides — the bracket/colon/position bytes corrupt
personalized DNA buffers in haplotype consumers (e.g. ``genvarloader``). Their
:func:`_symbolic_ilen` value is ``null`` (so they are also :data:`is_imprecise`).

To drop breakends, use ``expr=~genoray.exprs.is_breakend`` in a
:class:`genoray.Filter`. To drop *all* un-expandable ALTs (symbolic +
breakends) for haplotype consumers, combine::

    expr=~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend
"""


def _record_is_symbolic(alts: Iterable[str]) -> bool:
    """Record-level mirror of :data:`is_symbolic` for a cyvcf2 ``Variant.ALT``.

    True if any ALT allele is a symbolic allele (starts with ``<``). Used by the
    CLI to build the cyvcf2 record predicate half of a :class:`genoray.Filter`
    that must match its ``expr`` half on the VCF path.
    """
    return any(a.startswith("<") for a in alts)


def _record_is_breakend(alts: Iterable[str]) -> bool:
    """Record-level mirror of :data:`is_breakend` for a cyvcf2 ``Variant.ALT``.

    True if any ALT allele is a breakend (matches :data:`_BND_PATTERN`). Reuses
    the same regex as :data:`is_breakend` so a :class:`genoray.Filter`'s
    ``record`` and ``expr`` halves cannot drift apart.
    """
    return any(re.search(_BND_PATTERN, a) is not None for a in alts)


ILEN = pl.col("ALT").list.eval(pl.element().str.len_bytes().cast(pl.Int32)) - pl.col(
    "REF"
).str.len_bytes().cast(pl.Int32)
"""Indel length of the variant. Positive for insertions, negative for deletions, and zero for SNPs and MNPs."""


def _symbolic_ilen(
    alt: str = "ALT",
    ref: str = "REF",
    svlen: str = "SVLEN",
    end: str = "END",
    imprecise: str = "IMPRECISE",
) -> pl.Expr:
    """Per-ALT corrected ILEN as ``List[Int32]``.

    Non-symbolic, non-breakend ALTs use the literal ``len(ALT) - len(REF)``.
    Precise symbolic ``<DEL>``/``<INS>``/``<DUP>`` use ``SVLEN`` magnitude (or
    ``|END - POS|`` for ``<DEL>``/``<DUP>`` when ``SVLEN`` is absent):
    ``-|len|`` for ``<DEL>``, ``+|len|`` for ``<INS>``/``<DUP>``. Everything we
    cannot size precisely (``IMPRECISE`` flag, missing length, unsupported type
    such as ``<BND>``, ``<CNV>``, ``<INV>``, ``<*>``, and breakend mate notation
    such as ``G[chr2:321[``) becomes ``null``.

    ``svlen``/``end``/``imprecise`` name scalar columns already extracted by the
    caller (per record). ``SVLEN`` magnitude is read as ``|SVLEN|`` so the VCF
    4.3/4.4 sign-convention flip does not matter.

    .. note::
        Evaluated per-record via a Python callback (``map_elements``) at
        index-build time; expect one Python call per variant row.
    """
    ref_len = pl.col(ref).str.len_bytes().cast(pl.Int32)
    svlen_mag = pl.col(svlen).abs().cast(pl.Int32)
    end_mag = (pl.col(end) - pl.col("POS")).abs().cast(pl.Int32)
    mag = pl.coalesce(svlen_mag, end_mag)  # prefer SVLEN, fall back to END
    imprecise_flag = pl.col(imprecise).fill_null(False)

    # Extract the primary SV type per ALT element (e.g. "<DEL:ME>" -> "DEL").
    # list.eval only allows pl.element(); outer columns are broadcast automatically.
    sv_type_list = pl.col(alt).list.eval(
        pl.element().str.extract(r"^<([A-Za-z0-9:]+)>", 1).str.split(":").list.first()
    )
    # Literal (non-symbolic) ILEN per element: len(ALT_tok) - len(REF)
    literal_list = (
        pl.col(alt).list.eval(pl.element().str.len_bytes().cast(pl.Int32)) - ref_len
    )
    # Whether each ALT element is symbolic
    is_sym_list = pl.col(alt).list.eval(pl.element().str.starts_with("<"))
    # Whether each ALT element is a breakend (BND) — un-sizable like symbolic.
    is_bnd_list = pl.col(alt).list.eval(pl.element().str.contains(_BND_PATTERN))

    # Per-row scalar: sized SV length (sign-corrected), or null if un-sizable.
    # This is broadcast across all symbolic ALTs in the row.
    del_mag = pl.when(imprecise_flag | mag.is_null()).then(None).otherwise(-mag)
    ins_dup_mag = pl.when(imprecise_flag | mag.is_null()).then(None).otherwise(mag)

    # Build ILEN list element-wise using list.eval on a packed struct.
    # Pack: sv_type string + is_sym flag + literal value.
    packed = pl.struct(
        sv_type=sv_type_list,
        is_sym=is_sym_list,
        is_bnd=is_bnd_list,
        literal=literal_list,
        del_mag=del_mag,
        ins_dup_mag=ins_dup_mag,
    )

    return packed.map_elements(_compute_ilen_row, return_dtype=pl.List(pl.Int32))


def _compute_ilen_row(row: dict) -> list[int | None]:
    """Compute per-ALT ILEN for a single row (called via map_elements)."""
    sv_types: list = row["sv_type"]
    is_syms: list = row["is_sym"]
    is_bnds: list = row["is_bnd"]
    literals: list = row["literal"]
    del_mag = row["del_mag"]
    ins_dup_mag = row["ins_dup_mag"]

    result = []
    for sv_type, sym, bnd, lit in zip(sv_types, is_syms, is_bnds, literals):
        if bnd:
            # Breakends carry no expandable length — un-sizable, like symbolic SVs.
            result.append(None)
        elif not sym:
            result.append(lit)
        elif sv_type == "DEL":
            result.append(del_mag)
        elif sv_type in ("INS", "DUP"):
            result.append(ins_dup_mag)
        else:
            result.append(None)
    return result


is_imprecise = pl.col("ILEN").list.eval(pl.element().is_null()).list.any()
"""True if any ALT allele's ILEN could not be precisely determined (an un-sizable
symbolic allele — ``IMPRECISE``, missing ``SVLEN``/``END``, or an unsupported
symbolic type). Such alleles carry ``null`` ILEN. Filter them out with
``expr=~genoray.exprs.is_imprecise`` (in a :class:`genoray.Filter` for VCF, or
directly for PGEN) to keep precise structural variants while dropping the
rest; use ``~genoray.exprs.is_symbolic`` to drop *all* symbolic alleles
(required for haplotype consumers such as genvarloader, which cannot expand any
symbolic ALT)."""

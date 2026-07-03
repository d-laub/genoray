"""
Regression tests for polars ``DataFrame.join`` row-order assumptions.

``DataFrame.join`` / ``LazyFrame.join`` default to ``maintain_order='none'``,
which gives NO row-order guarantee: polars' multi-threaded hash join may
reorder the output relative to the left frame on large data, across runs, or
across polars versions. Small test data usually *looks* ordered, hiding the
bug. Any code that relies on the left frame's order after a join must pass
``maintain_order='left'`` explicitly.

These tests install a monkeypatch that *forces* the worst case: every join with
``maintain_order`` left at its ``'none'`` default reverses its output rows. Code
that correctly pins ``maintain_order='left'`` is unaffected; code that relies on
the implicit default breaks loudly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from genoray import SparseVar

ddir = Path(__file__).parent / "data"


@pytest.fixture
def reorder_unordered_joins(monkeypatch):
    """Force every ``maintain_order='none'`` join to reverse its output rows.

    This simulates polars' hash-join reordering deterministically so order
    bugs surface on small test data instead of silently in production.
    """
    orig_df_join = pl.DataFrame.join
    orig_lf_join = pl.LazyFrame.join

    def _is_unordered(maintain_order) -> bool:
        return maintain_order is None or maintain_order == "none"

    def df_join(self, other, *args, maintain_order=None, **kwargs):
        out = orig_df_join(self, other, *args, maintain_order=maintain_order, **kwargs)
        if _is_unordered(maintain_order):
            out = out.reverse()
        return out

    def lf_join(self, other, *args, maintain_order=None, **kwargs):
        out = orig_lf_join(self, other, *args, maintain_order=maintain_order, **kwargs)
        if _is_unordered(maintain_order):
            out = out.reverse()
        return out

    monkeypatch.setattr(pl.DataFrame, "join", df_join)
    monkeypatch.setattr(pl.LazyFrame, "join", lf_join)


def _gtf_covering_variants() -> pl.DataFrame:
    """A CDS GTF DataFrame overlapping every variant in the biallelic fixture.

    chr1 variants -> gene G1 (+ strand); chr2 variants -> gene G2 (- strand).
    """
    return pl.DataFrame(
        {
            "feature": ["CDS", "CDS"],
            "chrom": ["chr1", "chr2"],
            "start": [81000, 81000],
            "end": [82000, 82000],
            "strand": ["+", "-"],
            "frame": [0, 0],
            "gene_id": ["G1", "G2"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["1", "1"],
            "tag": ["canonical", "canonical"],
        }
    )


def test_annotate_with_gtf_preserves_index_order(tmp_path, reorder_unordered_joins):
    """``annotate_with_gtf(write_back=True)`` must not reorder ``self.index``.

    The variant index rows are positionally aligned with the sparse genotype
    storage: row ``i`` of ``self.index`` describes variant ``i``. The write-back
    join must preserve that order, regardless of how the hash join behaves.
    """
    src = ddir / "biallelic.vcf.svar"
    dst = tmp_path / "biallelic.vcf.svar"
    shutil.copytree(src, dst)

    svar = SparseVar(dst)
    original = svar.index.select("CHROM", "POS", "REF", "ALT")

    svar.annotate_with_gtf(_gtf_covering_variants(), level_filter=None, write_back=True)

    after = svar.index.select("CHROM", "POS", "REF", "ALT")
    assert after.equals(original), (
        "annotate_with_gtf reordered the variant index; row order no longer "
        "matches the sparse genotype storage."
    )

    # The annotation must also land on the correct contig.
    assert (
        svar.index.filter(pl.col("CHROM") == "chr1")["gene_id"].to_list() == ["G1"] * 3
    )
    assert (
        svar.index.filter(pl.col("CHROM") == "chr2")["gene_id"].to_list() == ["G2"] * 3
    )

"""
Unit tests for GTF annotation functions.

Tests the helper functions _empty_annot() and _get_strand_and_codon_pos()
without requiring full SVAR objects or files.
"""

from __future__ import annotations

import polars as pl
import pytest
from pytest_cases import case, parametrize_with_cases

from genoray._svar import _empty_annot, _get_strand_and_codon_pos
from genoray._utils import ContigNormalizer

GTF_SCHEMA = {
    "chrom": pl.Utf8,
    "start": pl.Int64,
    "end": pl.Int64,
    "strand": pl.Utf8,
    "frame": pl.Int64,
    "gene_id": pl.Utf8,
    "transcript_id": pl.Utf8,
    "gene_biotype": pl.Utf8,
    "transcript_support_level": pl.Utf8,
    "tag": pl.Utf8,
}


# Fixtures
@pytest.fixture
def basic_contig_normalizer():
    """Standard human chromosome normalizer."""
    return ContigNormalizer(["chr1", "chr2"])


# Helper Functions for Expected Values


def expected_codon_pos_plus(pos: int, cds_start: int, frame: int) -> int:
    """Positive strand: (rel_pos - frame) % 3, with 1-based coordinates"""
    return (pos - cds_start - frame) % 3


def expected_codon_pos_minus(pos: int, cds_start: int, frame: int) -> int:
    """Negative strand: (2 * (rel_pos - frame)) % 3, with 1-based coordinates"""
    return (2 * (pos - cds_start - frame)) % 3


# Test Cases for Parametrization


class CodonPositionCases:
    """Test cases covering all frame values for both strands."""

    @case(tags=["positive", "frame0"])
    def case_plus_frame0(self):
        """
        Positive strand, frame=0: codon positions are 0,1,2,0,1,2,...
        Position 1002 (rel=3): (3 + 0) % 3 = 0
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["+"],
                "frame": [0],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1003], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 0

    @case(tags=["positive", "frame1"])
    def case_plus_frame1(self):
        """
        Positive strand, frame=1: 1st base is at codon pos 2
        Position 1000 (rel=0): (0 + 2) % 3 = 2
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["+"],
                "frame": [1],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 2

    @case(tags=["positive", "frame2"])
    def case_plus_frame2(self):
        """
        Positive strand, frame=2: 1st base is at codon pos 1
        Position 1000 (rel=0): (0 + 1) % 3 = 1
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["+"],
                "frame": [2],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 1

    @case(tags=["negative", "frame0"])
    def case_minus_frame0(self):
        """
        Negative strand, frame=0: positions from start are 0,2,1,0,2,1,...
        Position 1002 (rel=2): (2 * 2) % 3 = 1
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["-"],
                "frame": [0],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1002], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 1

    @case(tags=["negative", "frame1"])
    def case_minus_frame1(self):
        """
        Negative strand, frame=1: 1st base is at codon pos 1
        Position 1000 (rel=0): (2 * (0 - 1)) % 3 = 1
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["-"],
                "frame": [1],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 1

    @case(tags=["negative", "frame2"])
    def case_minus_frame2(self):
        """
        Negative strand, frame=2: 1st base is at codon pos 2
        Position 1000 (rel=0): (2 * (0 - 2)) % 3 = 2
        """
        cds_df = pl.DataFrame(
            {
                "chrom": ["chr1"],
                "start": [1000],
                "end": [1009],
                "strand": ["-"],
                "frame": [2],
                "gene_id": ["G1"],
                "transcript_id": ["T1"],
                "gene_biotype": ["protein_coding"],
                "transcript_support_level": ["1"],
                "tag": ["canonical"],
            },
            schema=GTF_SCHEMA,
        )
        var_table = pl.DataFrame(
            {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
        )
        return cds_df, var_table, 2


# Basic Tests


def test_empty_annot_schema():
    """Property: _empty_annot() returns correct schema."""
    result = _empty_annot()
    assert result.shape == (0, 4)
    assert result.schema == {
        "varID": pl.UInt32,
        "gene_id": pl.Utf8,
        "strand": pl.Utf8,
        "codon_pos": pl.Int8,
    }


def test_empty_cds_returns_empty(basic_contig_normalizer):
    """Property: Empty CDS produces empty annotation."""
    cds_df = pl.DataFrame(
        schema={
            "chrom": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "strand": pl.Utf8,
            "frame": pl.Int64,
            "gene_id": pl.Utf8,
            "transcript_id": pl.Utf8,
            "gene_biotype": pl.Utf8,
            "transcript_support_level": pl.Utf8,
            "tag": pl.Utf8,
        }
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [101], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result.shape == (0, 4)


def test_no_overlap_returns_empty(basic_contig_normalizer):
    """Property: Non-overlapping variants produce empty annotation."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [1000],
            "end": [2000],
            "strand": ["+"],
            "frame": [0],
            "gene_id": ["G1"],
            "transcript_id": ["T1"],
            "gene_biotype": ["protein_coding"],
            "transcript_support_level": ["1"],
            "tag": ["canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [101], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result.shape == (0, 4)


# Codon Position Tests


@parametrize_with_cases("cds_df, var_table, expected", cases=CodonPositionCases)
def test_codon_position_all_frames(
    cds_df, var_table, expected, basic_contig_normalizer
):
    """Property: Codon position correctly calculated for all frame/strand combinations."""
    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)

    assert result.shape == (1, 4)
    assert result["codon_pos"][0] == expected


def test_indel_null_codon(basic_contig_normalizer):
    """Property: Indels receive null codon_pos."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [1000],
            "end": [1009],
            "strand": ["+"],
            "frame": [0],
            "gene_id": ["G1"],
            "transcript_id": ["T1"],
            "gene_biotype": ["protein_coding"],
            "transcript_support_level": ["1"],
            "tag": ["canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {
            "CHROM": ["chr1"],
            "POS": [1002],
            "ILEN": [[2]],
            "index": [0],  # 3bp insertion
        }
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result["codon_pos"][0] is None


# Transcript Ranking Tests


def test_ranking_protein_coding_wins(basic_contig_normalizer):
    """Property: Protein-coding beats non-coding, affecting codon_pos."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1009, 1009],
            "strand": ["+", "+"],
            "frame": [0, 1],
            "gene_id": ["G1", "G2"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "lncRNA"],
            "transcript_support_level": ["1", "1"],
            "tag": ["canonical", "canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result["gene_id"][0] == "G1"
    assert result["codon_pos"][0] == expected_codon_pos_plus(1000, 1000, 0)


def test_ranking_canonical_wins(basic_contig_normalizer):
    """Property: Canonical tag breaks ties, choosing its codon_pos."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1009, 1009],
            "strand": ["+", "+"],
            "frame": [0, 2],
            "gene_id": ["G1", "G1"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["1", "1"],
            "tag": ["canonical", None],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result["codon_pos"][0] == expected_codon_pos_plus(
        1000, 1000, 0
    )  # T1's frame


def test_ranking_tsl_wins(basic_contig_normalizer):
    """Property: Better TSL (lower number) wins, choosing its codon_pos."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1009, 1009],
            "strand": ["+", "+"],
            "frame": [0, 1],
            "gene_id": ["G1", "G1"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["3", "1"],
            "tag": [None, None],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    # T2 wins (TSL=1), has frame=1
    assert result["codon_pos"][0] == expected_codon_pos_plus(1000, 1000, 1)


def test_ranking_span_wins(basic_contig_normalizer):
    """Property: Longer CDS span wins when other factors equal."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1009, 1020],
            "strand": ["+", "+"],
            "frame": [0, 1],
            "gene_id": ["G1", "G1"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["1", "1"],
            "tag": [None, None],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    # T2 wins (longer span), has frame=1
    assert result["codon_pos"][0] == expected_codon_pos_plus(1000, 1000, 1)


def test_ranking_negative_strand_tsl(basic_contig_normalizer):
    """Property: Negative strand ranking also respects TSL."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [1000, 1000],
            "end": [1009, 1009],
            "strand": ["-", "-"],
            "frame": [0, 1],
            "gene_id": ["G1", "G1"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["2", "1"],
            "tag": [None, None],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1000], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    # T2 wins (TSL=1), has frame=1
    assert result["codon_pos"][0] == expected_codon_pos_minus(1000, 1000, 1)


# Multi-variant and Normalization Tests


def test_multiple_variants(basic_contig_normalizer):
    """Property: Multiple variants annotated independently."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chr2"],
            "start": [1000, 2000],
            "end": [1009, 2009],
            "strand": ["+", "-"],
            "frame": [0, 0],
            "gene_id": ["G1", "G2"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["1", "1"],
            "tag": ["canonical", "canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr2"],
            "POS": [1003, 2002],
            "ILEN": [[0], [0]],
            "index": [0, 1],
        }
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result.shape == (2, 4)
    assert list(result["gene_id"]) == ["G1", "G2"]


def test_chromosome_normalization():
    """Property: Chromosome names normalized (chr1 ↔ 1)."""
    normalizer = ContigNormalizer(["1"])
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "start": [1000],
            "end": [1009],
            "strand": ["+"],
            "frame": [0],
            "gene_id": ["G1"],
            "transcript_id": ["T1"],
            "gene_biotype": ["protein_coding"],
            "transcript_support_level": ["1"],
            "tag": ["canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["1"], "POS": [1003], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, normalizer)
    assert result.shape == (1, 4)
    assert result["gene_id"][0] == "G1"


def test_unmapped_chromosomes_filtered(basic_contig_normalizer):
    """Property: CDS on unmapped chromosomes are filtered."""
    cds_df = pl.DataFrame(
        {
            "chrom": ["chr1", "chrUn"],
            "start": [1000, 1000],
            "end": [1009, 1009],
            "strand": ["+", "+"],
            "frame": [0, 0],
            "gene_id": ["G1", "G2"],
            "transcript_id": ["T1", "T2"],
            "gene_biotype": ["protein_coding", "protein_coding"],
            "transcript_support_level": ["1", "1"],
            "tag": ["canonical", "canonical"],
        },
        schema=GTF_SCHEMA,
    )
    var_table = pl.DataFrame(
        {"CHROM": ["chr1"], "POS": [1003], "ILEN": [[0]], "index": [0]}
    )

    result = _get_strand_and_codon_pos(cds_df, var_table, basic_contig_normalizer)
    assert result.shape == (1, 4)
    assert result["gene_id"][0] == "G1"  # Only G1, not G2

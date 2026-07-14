from __future__ import annotations

import numpy as np
import polars as pl

from genoray._contigs import ContigNormalizer
from genoray._mutcat.strand import contig_strand_intervals, load_gene_intervals


def _genes() -> pl.DataFrame:
    # 0-based half-open gene footprints on chr1.
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [0, 30, 25],  # +[0,20), +[30,40), -[25,45)
            "stop": [20, 40, 45],
            "strand": ["+", "+", "-"],
        }
    )


def test_contig_intervals_partition_classes():
    c_norm = ContigNormalizer(["chr1"])
    starts, stops, values = contig_strand_intervals(_genes(), "chr1", c_norm)
    # Expected disjoint partition:
    #   [0,20)  + only  -> 1
    #   [25,30) - only  -> 2
    #   [30,40) both    -> 3
    #   [40,45) - only  -> 2
    assert starts.tolist() == [0, 25, 30, 40]
    assert stops.tolist() == [20, 30, 40, 45]
    assert values.tolist() == [1, 2, 3, 2]
    assert starts.dtype == np.int32
    assert values.dtype == np.uint8


def test_contig_name_normalization():
    # store contig is unprefixed "1"; GTF uses "chr1".
    c_norm = ContigNormalizer(["1"])
    starts, stops, values = contig_strand_intervals(_genes(), "1", c_norm)
    assert starts.size == 4


def test_no_genes_on_contig_is_empty():
    c_norm = ContigNormalizer(["chr2"])
    starts, stops, values = contig_strand_intervals(_genes(), "chr2", c_norm)
    assert starts.size == 0 and stops.size == 0 and values.size == 0


def test_load_gene_intervals_from_dataframe():
    gtf = pl.DataFrame(
        {
            "seqname": ["chr1", "chr1"],
            "feature": ["gene", "exon"],
            "start": [1, 5],  # 1-based
            "end": [40, 10],
            "strand": ["+", "+"],
        }
    )
    genes = load_gene_intervals(gtf)
    assert genes.height == 1  # only feature == "gene"
    row = genes.row(0, named=True)
    assert row["chrom"] == "chr1"
    assert row["start"] == 0  # 1-based 1 -> 0-based 0
    assert row["stop"] == 40  # inclusive 1-based 40 == exclusive 0-based 40
    assert row["strand"] == "+"

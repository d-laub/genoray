"""Build transcriptional-strand-class intervals from a GTF gene model.

Flattens gene footprints into a sorted, disjoint interval partition per contig,
each interval tagged with a coverage class (1=+only, 2=−only, 3=both strands).
Gaps between intervals are class N (nontranscribed) and are not materialized.
The result crosses the pyo3 boundary as a struct-of-arrays and drives the Rust
two-pointer strand sweep in ``annotate_contig``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import seqpro as sp
from numpy.typing import NDArray

from .._contigs import ContigNormalizer


def load_gene_intervals(gtf: str | pl.DataFrame) -> pl.DataFrame:
    """Load ``feature == "gene"`` footprints from a GTF as 0-based half-open rows.

    Parameters
    ----------
    gtf
        Path to a GTF/GTF.gz, or a pre-loaded Polars DataFrame with GTF columns
        (``seqname``/``chrom``, ``feature``, ``start`` [1-based], ``end``
        [inclusive], ``strand``).

    Returns
    -------
    pl.DataFrame
        Columns ``chrom`` (Utf8), ``start`` (Int64, 0-based), ``stop`` (Int64,
        exclusive), ``strand`` (Utf8, ``"+"``/``"-"``).
    """
    if isinstance(gtf, pl.DataFrame):
        df = gtf.rename({"seqname": "chrom"}, strict=False)
    else:
        df = sp.gtf.scan(str(gtf)).collect().rename({"seqname": "chrom"}, strict=False)
    return df.filter(pl.col("feature") == "gene").select(
        pl.col("chrom").cast(pl.Utf8),
        (pl.col("start") - 1).cast(pl.Int64).alias("start"),  # 1-based -> 0-based
        pl.col("end").cast(pl.Int64).alias("stop"),  # inclusive == exclusive 0-based
        pl.col("strand").cast(pl.Utf8),
    )


def _empty() -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]]:
    return (
        np.empty(0, np.int32),
        np.empty(0, np.int32),
        np.empty(0, np.uint8),
    )


def contig_strand_intervals(
    genes: pl.DataFrame, contig: str, c_norm: ContigNormalizer
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.uint8]]:
    """Flatten gene footprints on ``contig`` into a sorted, disjoint partition.

    Genes whose (normalized) chrom maps to ``contig`` are split by strand and
    merged into maximal runs, each tagged 1 (+only), 2 (−only), or 3 (both).
    Gaps (class N) are omitted. Returns ``(starts, stops, values)``.
    """
    normed = c_norm.norm(genes.get_column("chrom").to_list())
    g = genes.filter(pl.Series([n == contig for n in normed]))
    if g.height == 0:
        return _empty()

    ps = (
        g.filter(pl.col("strand") == "+")
        .get_column("start")
        .to_numpy()
        .astype(np.int64)
    )
    pe = (
        g.filter(pl.col("strand") == "+").get_column("stop").to_numpy().astype(np.int64)
    )
    ms = (
        g.filter(pl.col("strand") == "-")
        .get_column("start")
        .to_numpy()
        .astype(np.int64)
    )
    me = (
        g.filter(pl.col("strand") == "-").get_column("stop").to_numpy().astype(np.int64)
    )

    bp = np.unique(np.concatenate([ps, pe, ms, me]))
    if bp.size < 2:
        return _empty()

    def covered(
        starts: NDArray[np.int64], stops: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        # +1 at each start, -1 at each stop over the shared breakpoint grid; the
        # running sum on segment i = [bp[i], bp[i+1]) is that strand's coverage.
        diff = np.zeros(bp.size, dtype=np.int64)
        if starts.size:
            np.add.at(diff, np.searchsorted(bp, starts), 1)
            np.add.at(diff, np.searchsorted(bp, stops), -1)
        return np.cumsum(diff)[:-1] > 0

    pc = covered(ps, pe)
    mc = covered(ms, me)
    val = np.where(pc & mc, 3, np.where(pc, 1, np.where(mc, 2, 0))).astype(np.uint8)

    lo = bp[:-1]
    hi = bp[1:]
    genic = val != 0
    if not genic.any():
        return _empty()
    lo, hi, val = lo[genic], hi[genic], val[genic]

    # Merge adjacent segments that touch and share a class.
    starts_out = [int(lo[0])]
    stops_out = [int(hi[0])]
    vals_out = [int(val[0])]
    for i in range(1, val.size):
        if val[i] == vals_out[-1] and lo[i] == stops_out[-1]:
            stops_out[-1] = int(hi[i])
        else:
            starts_out.append(int(lo[i]))
            stops_out.append(int(hi[i]))
            vals_out.append(int(val[i]))

    return (
        np.asarray(starts_out, np.int32),
        np.asarray(stops_out, np.int32),
        np.asarray(vals_out, np.uint8),
    )

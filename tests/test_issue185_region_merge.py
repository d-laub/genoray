"""Tests for issue #185's region merge, which replaced a pyranges round-trip.

`_resolve_kept_rows` used `sp.bed.to_pyr(...).merge()` for two things only:
detecting whether any regions overlap, and collapsing them when the caller
allows it. `_merge_regions` does both in polars, so genoray no longer calls
pyranges at all -- and `pyranges` 0.1.x is what drags in `sorted-nearest`,
which publishes no wheels and so needs a C toolchain to install.

The merge rule has to match `PyRanges.merge()` (slack=0) exactly: the caller
compares row counts to decide whether to raise, so a rule that merged one case
more or less eagerly would change which region sets are rejected.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genoray._svar._regions import _merge_regions


def _bed(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"chrom": pl.Utf8, "start": pl.Int32, "end": pl.Int32},
        orient="row",
    )


def test_merge_regions_leaves_disjoint_regions_alone():
    """Height is the overlap signal, so untouched input must stay the same height."""
    regions = _bed([("c1", 0, 10), ("c1", 20, 30)])
    assert _merge_regions(regions).rows() == [("c1", 0, 10), ("c1", 20, 30)]


def test_merge_regions_merges_overlapping_regions():
    regions = _bed([("c1", 0, 15), ("c1", 10, 30)])
    assert _merge_regions(regions).rows() == [("c1", 0, 30)]


def test_merge_regions_merges_bookended_regions():
    """[0, 10) and [10, 20) touch without overlapping. `PyRanges.merge()` joins
    them, so genoray has always treated them as overlapping -- keep that."""
    regions = _bed([("c1", 0, 10), ("c1", 10, 20)])
    assert _merge_regions(regions).rows() == [("c1", 0, 20)]


def test_merge_regions_collapses_a_nested_region_into_its_container():
    """A running maximum, not a pairwise neighbour check: [25, 30) must still
    absorb [26, 28) even though the nested region ends first."""
    regions = _bed([("c1", 25, 30), ("c1", 26, 28)])
    assert _merge_regions(regions).rows() == [("c1", 25, 30)]


def test_merge_regions_does_not_merge_across_contigs():
    """Identical coordinates on different contigs are different regions."""
    regions = _bed([("c1", 0, 10), ("c2", 5, 15)])
    assert _merge_regions(regions).rows() == [("c1", 0, 10), ("c2", 5, 15)]


def test_merge_regions_preserves_the_bed_schema():
    merged = _merge_regions(_bed([("c1", 0, 10), ("c1", 5, 20)]))
    assert merged.columns == ["chrom", "start", "end"]
    assert merged.schema["start"] == pl.Int32
    assert merged.schema["end"] == pl.Int32


def test_merge_regions_matches_pyranges_on_random_region_sets():
    """Parity with the implementation this replaced, over inputs that mix
    overlap, nesting, bookending and zero-width intervals across contigs."""
    pr = pytest.importorskip("pyranges")
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 12))
        chrom = rng.choice(["c1", "c2", "c3"], size=n)
        start = rng.integers(0, 40, size=n)
        end = start + rng.integers(0, 12, size=n)

        mine = _merge_regions(
            pl.DataFrame(
                {
                    "chrom": chrom.tolist(),
                    "start": start.astype(np.int32),
                    "end": end.astype(np.int32),
                }
            )
        ).sort("chrom", "start", "end")

        gr = pr.PyRanges(
            pd.DataFrame({"Chromosome": chrom, "Start": start, "End": end})
        ).merge()
        theirs = (
            pl.from_pandas(gr.df)
            .rename({"Chromosome": "chrom", "Start": "start", "End": "end"})
            .with_columns(
                pl.col("chrom").cast(pl.Utf8),
                pl.col("start").cast(pl.Int32),
                pl.col("end").cast(pl.Int32),
            )
            .sort("chrom", "start", "end")
        )
        assert mine.equals(theirs)

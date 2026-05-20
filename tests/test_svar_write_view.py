from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from genoray._svar import _normalize_regions
from genoray._utils import ContigNormalizer


@pytest.fixture
def cnorm() -> ContigNormalizer:
    return ContigNormalizer(["chr1", "chr2"])


def test_normalize_regions_str(cnorm):
    df = _normalize_regions("chr1:10-20", cnorm)
    assert df.shape == (1, 3)
    assert df["chrom"].to_list() == ["chr1"]
    # 1-based inclusive "10-20" -> 0-based half-open [9, 20)
    assert df["start"].to_list() == [9]
    assert df["end"].to_list() == [20]


def test_normalize_regions_tuple(cnorm):
    df = _normalize_regions(("chr2", 5, 15), cnorm)
    assert df["chrom"].to_list() == ["chr2"]
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [15]


def test_normalize_regions_alt_contig_naming(cnorm):
    # "1" should normalize to "chr1"
    df = _normalize_regions(("1", 0, 100), cnorm)
    assert df["chrom"].to_list() == ["chr1"]


def test_normalize_regions_unknown_contig_dropped(cnorm):
    with pytest.warns(UserWarning, match="dropped"):
        df = _normalize_regions(("chrZZ", 0, 10), cnorm)
    assert df.height == 0


def test_normalize_regions_bed_file(tmp_path: Path, cnorm):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t100\t200\nchr2\t300\t400\n")
    df = _normalize_regions(bed, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [100, 300]
    assert df_sorted["end"].to_list() == [200, 400]


def test_normalize_regions_frame_polars_bio_schema(cnorm):
    frame = pl.DataFrame({"chrom": ["chr1"], "start": [5], "end": [25]})
    df = _normalize_regions(frame, cnorm)
    assert df["start"].to_list() == [5]
    assert df["end"].to_list() == [25]


def test_normalize_regions_pandas_frame(cnorm):
    import pandas as pd

    pdf = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [10, 20], "end": [30, 40]})
    df = _normalize_regions(pdf, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [10, 20]
    assert df_sorted["end"].to_list() == [30, 40]


def test_normalize_regions_pyranges(cnorm):
    pr = pytest.importorskip("pyranges")
    regions = pr.PyRanges(
        chromosomes=["chr1", "chr2"], starts=[100, 200], ends=[150, 250]
    )
    df = _normalize_regions(regions, cnorm)
    assert df.height == 2
    df_sorted = df.sort("chrom")
    assert df_sorted["chrom"].to_list() == ["chr1", "chr2"]
    assert df_sorted["start"].to_list() == [100, 200]
    assert df_sorted["end"].to_list() == [150, 250]


def test_normalize_regions_unsupported_type_raises(cnorm):
    with pytest.raises(TypeError, match="Unsupported regions type"):
        _normalize_regions(42, cnorm)  # type: ignore

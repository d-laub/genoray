from __future__ import annotations

import pytest

from genoray._cli._view_helpers import parse_regions_arg


def test_parse_regions_arg_single():
    df = parse_regions_arg("chr1:10-20")
    assert df["chrom"].to_list() == ["chr1"]
    # 1-based inclusive -> 0-based half-open
    assert df["start"].to_list() == [9]
    assert df["end"].to_list() == [20]


def test_parse_regions_arg_comma_list():
    df = parse_regions_arg("chr1:10-20,chr2:30-40")
    assert df["chrom"].to_list() == ["chr1", "chr2"]
    assert df["start"].to_list() == [9, 29]
    assert df["end"].to_list() == [20, 40]


def test_parse_regions_arg_bad_format():
    with pytest.raises(ValueError, match="region"):
        parse_regions_arg("not_a_region")

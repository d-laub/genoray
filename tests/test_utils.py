from __future__ import annotations

from pytest_cases import parametrize_with_cases

from variant_io._utils import ContigNormalizer


def contig_match():
    unnormed = "chr1"
    source = ContigNormalizer(["chr1", "chr2"])
    desired = "chr1"
    return unnormed, source, desired


def contig_add_match():
    unnormed = "1"
    source = ContigNormalizer(["chr1", "chr2"])
    desired = "chr1"
    return unnormed, source, desired


def contig_strip_match():
    unnormed = "chr1"
    source = ContigNormalizer(["1", "2"])
    desired = "1"
    return unnormed, source, desired


def contig_no_match():
    unnormed = "chr3"
    source = ContigNormalizer(["chr1", "chr2"])
    desired = None
    return unnormed, source, desired


def contig_list():
    unnormed = ["chr1", "1", "chr3"]
    source = ContigNormalizer(["chr1", "chr2"])
    desired = ["chr1", "chr1", None]
    return unnormed, source, desired


@parametrize_with_cases("unnormed, source, desired", cases=".", prefix="contig_")
def test_normalize_contig_name(
    unnormed: str | list[str], source: ContigNormalizer, desired: str | list[str] | None
):
    assert source.norm(unnormed) == desired

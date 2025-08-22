from __future__ import annotations

from pytest_cases import parametrize_with_cases

from genoray._utils import (
    ContigNormalizer,
    format_memory,
    parse_memory,
    variant_file_type,
)


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


def parse_b():
    mem = "32"
    desired = 32
    return mem, desired


def parse_k():
    mem = "1k"
    desired = 2**10
    return mem, desired


def parse_m():
    mem = "1m"
    desired = 2**20
    return mem, desired


def parse_g():
    mem = "32g"
    desired = 32 * 2**30
    return mem, desired


def parse_t():
    mem = "1t"
    desired = 2**40
    return mem, desired


def parse_p():
    mem = "1p"
    desired = 2**50
    return mem, desired


def parse_e():
    mem = "1e"
    desired = 2**60
    return mem, desired


def parse_kb():
    mem = "1kb"
    desired = 10**3
    return mem, desired


def parse_mb():
    mem = "1mb"
    desired = 10**6
    return mem, desired


def parse_gb():
    mem = "32gb"
    desired = 32 * 10**9
    return mem, desired


def parse_tb():
    mem = "1tb"
    desired = 10**12
    return mem, desired


def parse_pb():
    mem = "1pb"
    desired = 10**15
    return mem, desired


def parse_eb():
    mem = "1eb"
    desired = 10**18
    return mem, desired


@parametrize_with_cases("mem, desired", cases=".", prefix="parse_")
def test_parse_memory(mem: int | str, desired: int):
    assert parse_memory(mem) == desired


def format_b():
    mem = 1
    desired = "1 B"
    return mem, desired


def format_kb():
    mem = 2**10
    desired = "1.00 KiB"
    return mem, desired


def format_mb():
    mem = 2**20
    desired = "1.00 MiB"
    return mem, desired


def format_gb():
    mem = 2**30
    desired = "1.00 GiB"
    return mem, desired


def format_tb():
    mem = 2**40
    desired = "1.00 TiB"
    return mem, desired


def format_pb():
    mem = 2**50
    desired = "1.00 PiB"
    return mem, desired


def format_eb():
    mem = 2**60
    desired = "1.00 EiB"
    return mem, desired


@parametrize_with_cases("memory, desired", cases=".", prefix="format_")
def test_format_memory(memory: int, desired: str):
    assert format_memory(memory) == desired


def file_path_vcf():
    path = "test.vcf.gz"
    desired = "vcf"
    return path, desired


def file_path_pgen():
    path = "test.pgen"
    desired = "pgen"
    return path, desired


def file_path_vcf_preceding_dots():
    path = "hi.there.vcf.gz"
    desired = "vcf"
    return path, desired


def file_path_pgen_preceding_dots():
    path = "hi.there.pgen"
    desired = "pgen"
    return path, desired


@parametrize_with_cases("path, desired", cases=".", prefix="file_path_")
def test_variant_file_type(path: str, desired: str):
    assert variant_file_type(path) == desired

"""Regression tests for issue #139.

`VCF.get_record_info(info=[...])` used to lowercase the requested INFO names
before handing them to oxbow. oxbow matches INFO names case-sensitively against
the header's declared IDs, silently ignores the ones it does not recognize, and
drops its INFO column entirely when nothing matches -- so `info=["AF"]` became
`info_fields=["af"]`, matched nothing, and returned a frame with no INFO data
and no error. (The issue blamed `.rename(lambda c: c.upper())`; that node is
innocent -- the struct survives it intact -- and the bug does not need a
non-empty `fields=` list either.)

The fix resolves requested names against the declared IDs case-insensitively,
raises on names the header does not declare, and flattens the INFO struct into
top-level columns -- the shape the rest of genoray (`_fetch_info_cols`,
`_write_gvi_index`'s SVLEN/END/IMPRECISE handling) already speaks in.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
import pytest

from genoray import VCF

_ROWS = (
    "chr1\t10\t.\tA\tT\t.\t.\tAF=0.25;AC=1\tGT\t0|1\t0|0\n"
    "chr1\t20\t.\tC\tG\t.\t.\tAF=0.50;AC=2\tGT\t1|0\t0|1\n"
    "chr1\t30\t.\tG\tA\t.\t.\tAF=0.75;AC=3\tGT\t1|1\t0|1\n"
)


def _vcf_with_info(
    d: Path,
    *,
    info_header: str = (
        '##INFO=<ID=AF,Number=1,Type=Float,Description="Allele frequency">\n'
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">\n'
    ),
    rows: str = _ROWS,
) -> VCF:
    """A bgzipped, indexed two-sample VCF whose header declares INFO fields."""
    plain = d / "info.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        + info_header
        + '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n" + rows
    )
    gz = d / "info.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(plain)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", "-t", str(gz)], check=True)
    plain.unlink()
    return VCF(gz)


def test_get_record_info_returns_a_requested_info_field(tmp_path: Path):
    """The issue's exact repro: AF asked for, AF returned."""
    v = _vcf_with_info(tmp_path)
    df = v.get_record_info(fields=["CHROM", "POS", "REF", "ALT"], info=["AF"])
    assert "AF" in df.columns
    assert df.get_column("AF").to_list() == pytest.approx([0.25, 0.5, 0.75])


def test_get_record_info_returns_a_requested_info_field_without_a_fields_list(
    tmp_path: Path,
):
    """The issue claimed the bug needed a non-empty `fields=`. It did not --
    the lowercasing happens whether or not `fields` is None."""
    v = _vcf_with_info(tmp_path)
    df = v.get_record_info(info=["AF"])
    assert "AF" in df.columns
    assert df.get_column("AF").to_list() == pytest.approx([0.25, 0.5, 0.75])


def test_get_record_info_matches_info_names_case_insensitively(tmp_path: Path):
    """A mis-cased request resolves to the declared ID and is named by it, so
    the caller cannot be handed a column under a spelling no header uses."""
    v = _vcf_with_info(tmp_path)
    df = v.get_record_info(fields=["POS"], info=["af"])
    assert df.columns == ["POS", "AF"]


def test_get_record_info_rejects_an_undeclared_info_field(tmp_path: Path):
    """Silence is what made #139 expensive: oxbow ignores unknown names. Name
    both the offender and what the header actually declares."""
    v = _vcf_with_info(tmp_path)
    with pytest.raises(ValueError, match=r"NOPE.*not declared.*AF.*AC"):
        v.get_record_info(info=["NOPE"])


def test_get_record_info_flattens_every_declared_info_field_by_default(
    tmp_path: Path,
):
    """`info=None` still means all of them, but as columns -- not as a struct
    named after oxbow's internal layout."""
    v = _vcf_with_info(tmp_path)
    schema = v.get_record_info(lazy=True).collect_schema()
    assert "INFO" not in schema.names()
    assert schema["AF"] == pl.Float32
    assert schema["AC"] == pl.List(pl.Int32)


def test_get_record_info_returns_no_info_columns_for_an_empty_list(tmp_path: Path):
    """`info=[]` is the one request that legitimately yields no INFO."""
    v = _vcf_with_info(tmp_path)
    df = v.get_record_info(info=[])
    assert "AF" not in df.columns
    assert "AC" not in df.columns
    assert "INFO" not in df.columns


def test_get_record_info_reports_a_collision_with_a_non_info_column(tmp_path: Path):
    """An INFO ID may legally shadow a fixed column. Flattening then cannot be
    unambiguous, so say so instead of surfacing a polars duplicate-name error."""
    v = _vcf_with_info(
        tmp_path,
        info_header='##INFO=<ID=QUAL,Number=1,Type=Float,Description="shadow">\n',
        rows="chr1\t10\t.\tA\tT\t.\t.\tQUAL=0.25\tGT\t0|1\t0|0\n",
    )
    with pytest.raises(ValueError, match=r"QUAL.*collide"):
        v.get_record_info()


def test_get_record_info_honors_a_fields_list_without_chrom(tmp_path: Path):
    """CHROM is only cast when it was asked for; requesting other fields alone
    must not fail on a column the caller excluded."""
    v = _vcf_with_info(tmp_path)
    df = v.get_record_info(fields=["POS", "REF"], info=[])
    assert df.columns == ["POS", "REF"]


def test_write_gvi_index_keeps_a_requested_info_field(tmp_path: Path):
    """The downstream half of #139: the on-disk index silently lost any INFO
    column the caller asked for, which is what genvarloader works around."""
    v = _vcf_with_info(tmp_path)
    v._write_gvi_index(info=["AF"])
    index = pl.read_ipc(v._index_path())
    assert "AF" in index.columns
    assert index.get_column("AF").to_list() == pytest.approx([0.25, 0.5, 0.75])


def test_write_gvi_index_does_not_carry_unrequested_info(tmp_path: Path):
    """An index is a lookup table, not a copy of the annotations. It used to
    inherit oxbow's entire INFO struct -- which also defeated the rule below it
    that drops SVLEN/END/IMPRECISE the caller never asked for, since the struct
    kept its own copy of them."""
    v = _vcf_with_info(tmp_path)
    v._write_gvi_index()
    cols = pl.read_ipc(v._index_path()).columns
    assert "INFO" not in cols
    assert "AF" not in cols
    assert "AC" not in cols

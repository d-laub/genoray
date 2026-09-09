"""The `pipeline config` line must report every knob and where it came from.

On genoray 4.0.1 this line printed the planner's `reader_workers` while an
environment variable had set a different value, and a downstream operator spent
a day chasing the difference. With the environment channel gone the line is
correct by construction -- these tests hold it to reporting provenance too.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from genoray import SparseVar2, Tuning

pytest_plugins = ()

# 40 bp reference; the REF bases below match this exactly (1-based VCF POS):
# POS 3 = 'A', POS 7 = 'C', POS 12..14 = 'GTA'. Mirrors the pattern in
# tests/conftest.py's `svar2_store` fixture -- these tests only assert on a
# log line, not on conversion output, so a 3-record/2-sample/1-contig cohort
# is plenty. `small_vcf`/`small_pgen` are module-local (not added to the
# shared conftest.py): there is no `small_vcf`/`small_pgen` fixture upstream
# for this test module to reuse.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"

_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


@pytest.fixture(scope="module")
def small_vcf(tmp_path_factory) -> Path:
    """A tiny (3-record, 2-sample, single-contig) BCF+CSI for banner tests."""
    d = tmp_path_factory.mktemp("banner-vcf")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)
    return bcf


@pytest.fixture(scope="module")
def small_pgen(tmp_path_factory) -> Path:
    """The same 3-record, 2-sample, single-contig cohort as `small_vcf`, as a PGEN."""
    d = tmp_path_factory.mktemp("banner-pgen")
    vcf = d / "in.vcf"
    vcf.write_text(_VCF)
    gz = d / "in.vcf.gz"
    with open(gz, "wb") as fh:
        subprocess.run(["bgzip", "-c", str(vcf)], check=True, stdout=fh)
    subprocess.run(["bcftools", "index", str(gz)], check=True)
    subprocess.run(
        [
            "plink2",
            "--make-pgen",
            "--output-chr",
            "chrM",
            "--vcf",
            str(gz),
            "--out",
            str(d / "in"),
        ],
        check=True,
    )
    return d / "in.pgen"


def _banner(caplog_text: str) -> dict[str, str]:
    """Parse the flat `key=value` pairs out of the `pipeline config` line."""
    for line in caplog_text.splitlines():
        if "pipeline config" not in line:
            continue
        return dict(re.findall(r"(\w+)=([^\s]+)", line))
    raise AssertionError(f"no `pipeline config` line found in:\n{caplog_text}")


# `_svar2.py` does not yet pass `tuning=`/`log_filter=` through to `_core` --
# that is Task 7 (see `.superpowers/sdd/2026-09-08-explicit-tuning-api/
# task-6-rulings.md`, Ruling E). All three tests below call `from_vcf`/
# `from_pgen` with one or both of those keywords, so all three currently fail
# with `TypeError: ... got an unexpected keyword argument`. `strict=True` so
# Task 7 is forced to remove these markers rather than inheriting a
# permanently-xfail test.
_XFAIL_UNTIL_TASK_7 = pytest.mark.xfail(
    reason="from_vcf/from_pgen do not accept tuning=/log_filter= until Task 7 "
    "wires them through from Python",
    strict=True,
)


@_XFAIL_UNTIL_TASK_7
def test_explicit_values_are_tagged_explicit(tmp_path, small_vcf, capsys):
    out = tmp_path / "explicit.svar"
    SparseVar2.from_vcf(
        out,
        small_vcf,
        no_reference=True,
        tuning=Tuning(reader_workers=2, overshard=3, dense_cap=7),
        log_filter="genoray=info",
    )
    fields = _banner(capsys.readouterr().err)
    assert fields["reader_workers"] == "2"
    assert fields["reader_workers_src"] == "explicit"
    assert fields["overshard"] == "3"
    assert fields["overshard_src"] == "explicit"
    assert fields["dense_cap"] == "7"
    assert fields["dense_cap_src"] == "explicit"


@_XFAIL_UNTIL_TASK_7
def test_unset_values_are_tagged_planner(tmp_path, small_vcf, capsys):
    out = tmp_path / "planner.svar"
    SparseVar2.from_vcf(out, small_vcf, no_reference=True, log_filter="genoray=info")
    fields = _banner(capsys.readouterr().err)
    for knob in (
        "concurrent_chroms",
        "reader_workers",
        "overshard",
        "dense_cap",
        "merge_threads",
        "sample_interval",
    ):
        assert fields[f"{knob}_src"] == "planner", knob


@_XFAIL_UNTIL_TASK_7
def test_banner_omits_knobs_the_backend_cannot_use(tmp_path, small_pgen, capsys):
    out = tmp_path / "pgen.svar"
    SparseVar2.from_pgen(out, small_pgen, no_reference=True, log_filter="genoray=info")
    fields = _banner(capsys.readouterr().err)
    # from_pgen pins P=1 and never shards within a contig, so advertising these
    # would be advertising an inert knob.
    assert "overshard" not in fields
    assert "reader_workers_src" not in fields
    assert fields["concurrent_chroms_src"] == "planner"

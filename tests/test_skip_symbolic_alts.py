"""Tests for the ``skip_symbolic_alts`` opt-in filter on VCF and SparseVar.

These tests construct a tiny VCF with a mix of precise SNV/indel records and
VCF 4.x symbolic alleles (``<DEL>``, ``<INS:ME:ALU>``) using ``vcfixture``,
then assert that:

* ``VCF(..., skip_symbolic_alts=False)`` (default) keeps all 4 records.
* ``VCF(..., skip_symbolic_alts=True)`` drops the 2 symbolic records, keeping
  only the precise SNV/indel.
* ``SparseVar.from_vcf(..., skip_symbolic_alts=True)`` writes an SVAR whose
  ``index`` reflects only the precise records (i.e. the per-contig genotype
  scan and the index agree).
* ``genoray.exprs.is_symbolic`` evaluates as expected against the .gvi schema.
* Default behavior is preserved when the flag is omitted (regression guard).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from vcfixture import VcfBuilder

from genoray import VCF, SparseVar
from genoray import exprs as gexprs


def _build_mixed_vcf(tmp_path: Path) -> Path:
    """Render and index a tiny mixed precise + symbolic VCF.

    Layout::

        chr1  100  A      T          (SNV)
        chr1  200  A      <DEL>      (symbolic SV)
        chr1  300  C      <INS:ME:ALU> (symbolic SV)
        chr1  400  G      GAT        (precise insertion)
    """
    b = VcfBuilder(
        samples=["s1", "s2"],
        contigs=[("chr1", None)],
        fileformat="VCFv4.2",
    ).fmt("GT")
    b.record("chr1", 100, ref="A", alt=["T"], gt=["0|1", "1|1"])
    b.record("chr1", 200, ref="A", alt=["<DEL>"], gt=["0|1", "0|0"])
    b.record("chr1", 300, ref="C", alt=["<INS:ME:ALU>"], gt=["0|0", "0|1"])
    b.record("chr1", 400, ref="G", alt=["GAT"], gt=["1|1", "0|1"])
    vcf_path = tmp_path / "mixed.vcf"
    b.write(vcf_path)
    subprocess.run(["bgzip", "-f", str(vcf_path)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_path) + ".gz"], check=True)
    return Path(str(vcf_path) + ".gz")


def test_vcf_default_keeps_symbolic(tmp_path: Path) -> None:
    """Regression: default behavior must include symbolic records."""
    vcf_path = _build_mixed_vcf(tmp_path)
    vcf = VCF(vcf_path, with_gvi_index=False)
    vcf._write_gvi_index()
    vcf._load_index()
    assert vcf._index is not None
    assert vcf._index.height == 4
    alts = vcf._index["ALT"].to_list()
    assert ["<DEL>"] in alts
    assert ["<INS:ME:ALU>"] in alts


def test_vcf_skip_symbolic_drops_symbolic(tmp_path: Path) -> None:
    """``skip_symbolic_alts=True`` must drop both ``<DEL>`` and ``<INS:ME:ALU>``."""
    vcf_path = _build_mixed_vcf(tmp_path)
    # Pre-build the unfiltered index so _load_index has something to scan.
    VCF(vcf_path, with_gvi_index=False)._write_gvi_index()

    vcf = VCF(vcf_path, skip_symbolic_alts=True)
    assert vcf._skip_symbolic_alts is True
    assert vcf._filter is not None
    assert vcf._pl_filter is not None

    # When the *only* filter is the symbolic-ALT skip the constructor
    # auto-loads the index, because the cyvcf2 lambda and polars expr
    # are paired by construction. (Bare ``filter=...`` users have to
    # call _load_index themselves — see test_skip_symbolic_composes_with_user_filter.)
    assert vcf._index is not None
    assert vcf._index.height == 2
    refs = vcf._index["REF"].to_list()
    alts = [a[0] for a in vcf._index["ALT"].to_list()]
    assert sorted(zip(refs, alts)) == [("A", "T"), ("G", "GAT")]


def test_vcf_skip_symbolic_paired_filter_round_trip(tmp_path: Path) -> None:
    """The cyvcf2 filter must agree with the polars filter (records must be
    filtered identically when iterating via cyvcf2)."""
    import cyvcf2

    vcf_path = _build_mixed_vcf(tmp_path)
    VCF(vcf_path, with_gvi_index=False)._write_gvi_index()

    vcf = VCF(vcf_path, skip_symbolic_alts=True)
    assert vcf._filter is not None

    raw = cyvcf2.VCF(str(vcf_path))
    kept = [r.POS for r in raw if vcf._filter(r)]
    assert kept == [100, 400]


def test_is_symbolic_expression(tmp_path: Path) -> None:
    """``genoray.exprs.is_symbolic`` should yield True iff any ALT starts with '<'."""
    df = pl.DataFrame(
        {
            "ALT": [["T"], ["<DEL>"], ["G", "<INS>"], ["GAT"], ["<INS:ME:ALU>"]],
        }
    )
    got = df.select(gexprs.is_symbolic.alias("sym"))["sym"].to_list()
    assert got == [False, True, True, False, True]


def test_svar_from_vcf_skip_symbolic_alts(tmp_path: Path) -> None:
    """``SparseVar.from_vcf(skip_symbolic_alts=True)`` must produce an SVAR
    whose index is restricted to precise records.

    This is the load-bearing assertion: the .gvi index *and* the per-contig
    sparse genotype scan must agree on the filtered record set, otherwise
    downstream haplotype injection over-indexes and the SVAR is corrupt.
    """
    vcf_path = _build_mixed_vcf(tmp_path)
    out = tmp_path / "mixed.svar"
    if out.exists():
        shutil.rmtree(out)

    vcf = VCF(vcf_path, skip_symbolic_alts=True)
    SparseVar.from_vcf(out, vcf, max_mem="64m", overwrite=True)

    svar = SparseVar(out)
    # 4 input records, 2 symbolic → 2 in the SVAR.
    assert svar.index.height == 2
    refs = svar.index["REF"].to_list()
    alts = [a[0] for a in svar.index["ALT"].to_list()]
    assert sorted(zip(refs, alts)) == [("A", "T"), ("G", "GAT")]


def test_svar_from_vcf_default_keeps_symbolic(tmp_path: Path) -> None:
    """Regression: ``from_vcf`` default must not filter (back-compat)."""
    vcf_path = _build_mixed_vcf(tmp_path)
    out = tmp_path / "mixed_default.svar"
    if out.exists():
        shutil.rmtree(out)
    SparseVar.from_vcf(out, VCF(vcf_path), max_mem="64m", overwrite=True)
    svar = SparseVar(out)
    assert svar.index.height == 4


def test_svar_from_vcf_kwarg_overrides_vcf(tmp_path: Path) -> None:
    """An explicit ``skip_symbolic_alts=True`` on ``from_vcf`` should take
    effect even when the source VCF was constructed without the flag."""
    vcf_path = _build_mixed_vcf(tmp_path)
    out = tmp_path / "mixed_override.svar"
    if out.exists():
        shutil.rmtree(out)

    vcf = VCF(vcf_path)  # no skip_symbolic_alts
    assert vcf._skip_symbolic_alts is False
    SparseVar.from_vcf(out, vcf, max_mem="64m", overwrite=True, skip_symbolic_alts=True)
    svar = SparseVar(out)
    assert svar.index.height == 2


def test_svar_from_vcf_kwarg_rejects_user_filter_conflict(tmp_path: Path) -> None:
    """If a user provided a custom cyvcf2 ``filter`` and asks us to override
    with ``skip_symbolic_alts=True``, we should raise rather than silently
    drop their filter."""
    vcf_path = _build_mixed_vcf(tmp_path)
    out = tmp_path / "mixed_conflict.svar"

    vcf = VCF(
        vcf_path,
        filter=lambda v: True,
        pl_filter=pl.lit(True),
    )
    with pytest.raises(ValueError, match="skip_symbolic_alts=True conflicts"):
        SparseVar.from_vcf(
            out, vcf, max_mem="64m", overwrite=True, skip_symbolic_alts=True
        )


def test_skip_symbolic_composes_with_user_filter(tmp_path: Path) -> None:
    """When a user passes both a ``filter`` and ``skip_symbolic_alts=True``
    to ``VCF()``, both must apply (AND semantics)."""
    vcf_path = _build_mixed_vcf(tmp_path)
    VCF(vcf_path, with_gvi_index=False)._write_gvi_index()

    # Keep only records where REF == "A". With skip_symbolic_alts this must
    # leave POS=100 only (POS=200 is <DEL>, which is REF=A but symbolic).
    vcf = VCF(
        vcf_path,
        filter=lambda v: v.REF == "A",
        pl_filter=pl.col("REF") == "A",
        skip_symbolic_alts=True,
    )
    vcf._load_index()
    assert vcf._index is not None
    assert vcf._index["POS"].to_list() == [100]

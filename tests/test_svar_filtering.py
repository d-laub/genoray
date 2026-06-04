"""Filter-inheritance tests for VCF/PGEN -> SVAR.

Replaces the old skip_symbolic_alts flag tests: symbolic filtering is now just
`pl_filter=~exprs.is_symbolic` (+ paired cyvcf2 `filter`), and SparseVar inherits
the source's filter.
"""

from __future__ import annotations

from pathlib import Path

from vcfixture import Seq, Sym, VcfBuilder, VcfVersion

from genoray import VCF
from genoray import exprs as gexprs


def _not_symbolic(rec) -> bool:
    return not any(a.startswith("<") for a in rec.ALT)


def _mixed_vcf(tmp_path: Path) -> Path:
    """chr1: SNV A>T@100, <DEL>@200, <INS>@300, ins G>GAT@400."""
    b = (
        VcfBuilder(
            samples=["s1", "s2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .info("SVLEN")
        .info("SVCLAIM")
        .info("END")
        .fmt("GT")
    )
    b.record("chr1", 100, ref="A", alt=[Seq("T")], gt=["0|1", "1|1"])
    b.record(
        "chr1",
        200,
        ref="A",
        alt=[Sym.deletion()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [50], "SVCLAIM": ["D"], "END": [250]},
    )
    b.record(
        "chr1",
        300,
        ref="C",
        alt=[Sym.insertion()],
        gt=["0|0", "0|1"],
        info={"SVLEN": [60]},
    )
    b.record("chr1", 400, ref="G", alt=[Seq("GAT")], gt=["1|1", "0|1"])
    return b.write(tmp_path / "mixed.vcf.gz", bgzip=True, index=True)


def test_vcf_load_index_list_typed_filter(tmp_path):
    """A list-typed pl_filter (is_symbolic) must evaluate on the VCF path."""
    vcf_path = _mixed_vcf(tmp_path)
    v = VCF(vcf_path, filter=_not_symbolic, pl_filter=~gexprs.is_symbolic)
    v._write_gvi_index()
    v._load_index()
    assert v._index is not None
    assert v._index.height == 2
    assert v._index["POS"].to_list() == [100, 400]

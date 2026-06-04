"""Single source of truth for genoray test VCFs.

Each function returns a vcfixture.VcfBuilder reproducing the exact contig/pos/
ref/alt/GT/DS content of the formerly-tracked .vcf files. `gen_vcfs.py` renders
these to tests/data/<name>.vcf; tests import the same builders to obtain the
decoded GroundTruth oracle.
"""

from __future__ import annotations

from vcfixture import Number, Seq, Sym, Type, VcfBuilder, VcfVersion

NAN = float("nan")


def biallelic() -> VcfBuilder:
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None), ("chr2", None), ("chr3", None)],
            version=VcfVersion.V4_1,
        )
        .fmt("GT")
        .fmt("DS", Number.A, Type.FLOAT)
    )
    b.record(
        "chr1", 81262, ref="GAT", alt=[Seq("A")], gt=["0|1", "1|1"], DS=[[1.0], [2.0]]
    )
    b.record("chr1", 81262, ref="G", alt=[Seq("A")], gt=["./.", "0/1"], DS=[NAN, [1.0]])
    b.record("chr1", 81265, ref="T", alt=[Seq("C")], gt=["1|0", "./."], DS=[[0.9], NAN])
    b.record(
        "chr2", 81262, ref="GAT", alt=[Seq("A")], gt=["0|0", "1|1"], DS=[[0.0], [2.0]]
    )
    b.record(
        "chr2", 81262, ref="G", alt=[Seq("A")], gt=["./1", "0/1"], DS=[[1.0], [1.0]]
    )
    b.record("chr2", 81265, ref="T", alt=[Seq("C")], gt=["1|0", "./."], DS=[[0.9], NAN])
    return b


def multiallelic() -> VcfBuilder:
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None)],
            version=VcfVersion.V4_1,
        )
        .fmt("GT")
        .fmt("DS", Number.A, Type.FLOAT)
    )
    b.record(
        "chr1", 81262, ref="GAT", alt=[Seq("A")], gt=["0|1", "1|1"], DS=[[1.0], [2.0]]
    )
    b.record(
        "chr1",
        81262,
        ref="G",
        alt=[Seq("A"), Seq("C")],
        gt=["./.", "0/2"],
        DS=[NAN, [NAN, 1.0]],
    )
    return b


def three_samples_unsorted() -> VcfBuilder:
    b = VcfBuilder(
        samples=["sample_C", "sample_A", "sample_B"],
        contigs=[("chr1", 200)],
        version=VcfVersion.V4_2,
    ).fmt("GT")
    b.record("chr1", 100, ref="T", alt=[Seq("A")], gt=["0|1", "1|1", "0|0"])
    return b


def indels() -> VcfBuilder:
    """with_length edge-case fixture. POS preserved verbatim.

    Region A (1000): -10 deletion + 6 SNPs. Region B (2000): -4 deletion het on
    sample1 hapA only + 3 SNPs. Region C (3000): -30 deletion + 40 dense SNPs.
    Region D (5000): lone -10 deletion as last variant. Not a realistic genome.
    """
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", None)],
            version=VcfVersion.V4_2,
        )
        .fmt("GT")
        .fmt("DS", Number.ONE, Type.FLOAT)
    )

    def snp(pos: int, gt: list[str], ds: list[float]) -> None:
        b.record("chr1", pos, ref="T", alt=[Seq("C")], gt=gt, DS=ds)

    b.record(
        "chr1",
        1000,
        ref="G" + "A" * 10,
        alt=[Seq("G")],
        gt=["1|1", "1|1"],
        DS=[2.0, 2.0],
    )
    for p in range(1011, 1022, 2):
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    b.record(
        "chr1",
        2000,
        ref="G" + "A" * 4,
        alt=[Seq("G")],
        gt=["1|0", "0|0"],
        DS=[1.0, 0.0],
    )
    for p in (2002, 2004, 2006):
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    b.record(
        "chr1",
        3000,
        ref="G" + "A" * 30,
        alt=[Seq("G")],
        gt=["1|1", "1|1"],
        DS=[2.0, 2.0],
    )
    for p in range(3031, 3071):
        snp(p, ["1|1", "1|1"], [2.0, 2.0])

    b.record(
        "chr1",
        5000,
        ref="G" + "A" * 10,
        alt=[Seq("G")],
        gt=["1|1", "1|1"],
        DS=[2.0, 2.0],
    )
    return b


def symbolic() -> VcfBuilder:
    """Symbolic structural variants for ILEN correctness tests.

    Precise <DEL>/<INS>/<DUP> (SVLEN/SVCLAIM), then un-sizable cases:
    an IMPRECISE <DEL> and a <CNV> with no usable SVLEN. VCF 4.4 so SVLEN is
    positive and <DEL>/<DUP> carry SVCLAIM. (A <BND> row is added in Task 6.)
    """
    b = (
        VcfBuilder(
            samples=["sample1", "sample2"],
            contigs=[("chr1", 1_000_000)],
            version=VcfVersion.V4_4,
        )
        .fmt("GT")
        .info("SVLEN")
        .info("END")
        .info("SVCLAIM")
        .info("IMPRECISE")
    )
    b.record(
        "chr1",
        1000,
        ref="G",
        alt=[Sym.deletion()],
        gt=["0|1", "1|1"],
        info={"SVLEN": [100], "END": [1100], "SVCLAIM": ["D"]},
    )
    b.record(
        "chr1",
        2000,
        ref="G",
        alt=[Sym.insertion()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [50]},
    )
    b.record(
        "chr1",
        3000,
        ref="G",
        alt=[Sym.duplication()],
        gt=["1|0", "1|1"],
        info={"SVLEN": [30], "END": [3030], "SVCLAIM": ["DJ"]},
    )
    b.record(
        "chr1",
        4000,
        ref="G",
        alt=[Sym.deletion()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [200], "END": [4200], "SVCLAIM": ["D"], "IMPRECISE": True},
    )
    b.record(
        "chr1",
        5000,
        ref="G",
        alt=[Sym.cnv()],
        gt=["0|1", "0|0"],
        info={"SVLEN": [40], "END": [5040], "SVCLAIM": ["D"]},
    )
    return b


FIXTURES = {
    "biallelic": biallelic,
    "multiallelic": multiallelic,
    "three_samples_unsorted": three_samples_unsorted,
    "indels": indels,
    "symbolic": symbolic,
}

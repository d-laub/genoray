"""Tests for SparseVar2.write_view (region/sample subset via re-conversion)."""

import hashlib
import subprocess
from pathlib import Path

import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2
from genoray._svar2_fields import InfoField


def _dir_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }


def test_write_view_reroute_false_not_implemented(svar2_store, tmp_path):
    sv = SparseVar2(svar2_store)
    with pytest.raises(NotImplementedError, match="reroute"):
        sv.write_view(
            (sv.contigs[0], 0, 40),
            sv.available_samples,
            tmp_path / "v.svar2",
            reroute=False,
        )


def test_write_view_self_overwrite_guard(svar2_store):
    sv = SparseVar2(svar2_store)
    with pytest.raises(ValueError, match="in place|same path"):
        sv.write_view(
            (sv.contigs[0], 0, 40),
            sv.available_samples,
            sv.path,
            overwrite=True,
        )


def test_write_view_byte_parity_with_from_vcf(tmp_path):
    """reroute=True on a full region+all samples == a fresh from_vcf on the same input.

    Self-contained oracle: builds its own single-contig VCF+ref (no fields, no
    signatures) rather than reusing the session `svar2_store` fixture, which
    exposes no vcf/ref paths. A full-region/all-sample view re-runs the same
    cost model on the same effective variants with no stored fields/signatures
    to carry through, so the routed sidecar bytes should be identical.
    """
    ref_seq = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{ref_seq}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=40>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\n"
        "chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\n"
        "chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\n"
        "chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\n"
    )
    vcf_gz = tmp_path / "in.vcf.gz"
    subprocess.run(f"bgzip -c {vcf} > {vcf_gz}", shell=True, check=True)
    subprocess.run(["bcftools", "index", str(vcf_gz)], check=True)

    direct = tmp_path / "direct.svar2"
    SparseVar2.from_vcf(direct, vcf_gz, ref, threads=1, overwrite=True)
    direct_sv = SparseVar2(direct)

    viewed = tmp_path / "viewed.svar2"
    direct_sv.write_view(
        (direct_sv.contigs[0], 0, len(ref_seq)),
        direct_sv.available_samples,
        viewed,
        overwrite=True,
    )
    viewed_sv = SparseVar2(viewed)

    assert viewed_sv.contigs == direct_sv.contigs
    assert viewed_sv.available_samples == direct_sv.available_samples
    for c in direct_sv.contigs:
        assert _dir_digest(direct / c) == _dir_digest(viewed / c)


def test_write_view_default_fields_is_genotypes_only(tmp_path):
    """`write_view`'s default `fields=None` must mean "no fields" (genotypes
    only), which must succeed on ANY store -- including one built with
    `info_fields=`/`format_fields=`.

    Regresses a bug where `write_view` computed `fields_to_write` via
    `_validate_fields(fields, available)`, whose `None` semantics are "all
    available fields" (the read-path convention), not "no fields". On a store
    with a field present, that expanded the default to a non-empty list,
    which the Rust backend rejects outright ("field carry-through is not yet
    implemented for SVAR2 views") -- so a plain `write_view(...)` call with no
    `fields=` argument hard-failed on any store with fields, contradicting the
    documented "genotypes only" default.
    """
    doc = (
        VcfBuilder(samples=["s1", "s2"], contigs=[("chr1", None)])
        .fmt("GT")
        .info("AF", Number.A, Type.FLOAT)
        .record(
            "chr1",
            1000,
            ref="A",
            alt=[Seq("T")],
            gt=["0|1", "1|1"],
            info={"AF": [0.25]},
        )
    )
    vcf = doc.write(tmp_path / "src.vcf.gz", bgzip=True, index=True)

    src = tmp_path / "fields.svar2"
    SparseVar2.from_vcf(
        src,
        vcf,
        no_reference=True,
        info_fields=[InfoField("AF", dtype="f32")],
    )
    sv = SparseVar2(src)
    assert sv.available_fields  # sanity: the field is really there

    # Default `fields=` -> genotypes only -> must succeed even though the
    # source store has a field.
    default_out = tmp_path / "default.svar2"
    sv.write_view(
        ("chr1", 0, 2000),
        sv.available_samples,
        default_out,
        overwrite=True,
    )
    viewed = SparseVar2(default_out)
    assert viewed.contigs == sv.contigs
    assert viewed.available_samples == sv.available_samples

    # Explicitly requesting the field is the documented not-yet-implemented
    # path: it must still raise.
    explicit_out = tmp_path / "explicit.svar2"
    with pytest.raises(ValueError, match="field carry-through"):
        sv.write_view(
            ("chr1", 0, 2000),
            sv.available_samples,
            explicit_out,
            fields=["AF"],
            overwrite=True,
        )

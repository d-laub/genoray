"""Tests for SparseVar2.write_view (region/sample subset via re-conversion, and
the reroute=False direct-slice path)."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2, _core
from genoray._svar2_fields import FormatField, InfoField

_REF_SEQ = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"


def _dir_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }


def _read_if_exists(p: Path) -> bytes:
    return p.read_bytes() if p.exists() else b""


def test_write_view_reroute_false_smoke(svar2_store, tmp_path):
    """reroute=False now succeeds and produces a readable store (previously
    NotImplementedError)."""
    sv = SparseVar2(svar2_store)
    out = tmp_path / "v.svar2"
    sv.write_view(
        (sv.contigs[0], 0, 40),
        sv.available_samples,
        out,
        reroute=False,
    )
    viewed = SparseVar2(out)
    assert viewed.contigs == sv.contigs
    assert viewed.available_samples == sv.available_samples
    # non-vacuity: the view actually has decodable content
    rag = viewed.decode(sv.contigs[0], [(0, 40)])
    assert rag["pos"].lengths.sum() > 0


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

    # Explicitly requesting the field on the reroute=True (default "auto")
    # path is the documented not-yet-implemented path: it must still raise.
    explicit_out = tmp_path / "explicit.svar2"
    with pytest.raises(ValueError, match="field carry-through"):
        sv.write_view(
            ("chr1", 0, 2000),
            sv.available_samples,
            explicit_out,
            fields=["AF"],
            overwrite=True,
        )


def test_reroute_false_equivalent_to_true(svar2_store, tmp_path):
    """reroute=True vs reroute=False must decode to the SAME genotypes on a
    real sample subset (representation on disk may differ; decoded content
    must not).

    NOTE: this compares genotypes only (`fields=None`). `reroute=True` rejects
    non-empty `fields` (field carry-through is only implemented for
    `reroute=False` -- see `test_reroute_false_carries_fields...` below), so a
    fields= equivalence test isn't possible on the `reroute=True` side.
    """
    sv = SparseVar2(svar2_store)
    contig = sv.contigs[0]
    samples = sv.available_samples[:1]

    a = tmp_path / "true.svar2"
    b = tmp_path / "false.svar2"
    sv.write_view((contig, 0, 40), samples, a, reroute=True, overwrite=True)
    sv.write_view((contig, 0, 40), samples, b, reroute=False, overwrite=True)

    ra = SparseVar2(a).decode(contig, [(0, 40)])
    rb = SparseVar2(b).decode(contig, [(0, 40)])
    assert ra.shape == rb.shape
    for key in ("pos", "ilen", "allele"):
        np.testing.assert_array_equal(
            np.asarray(ra[key].data), np.asarray(rb[key].data)
        )
        np.testing.assert_array_equal(ra[key].lengths, rb[key].lengths)


def _fields_vcf_and_ref(tmp_path: Path) -> tuple[Path, Path]:
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF_SEQ}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    doc = (
        VcfBuilder(samples=["S0", "S1"], contigs=[("chr1", 40)])
        .info("AF", Number.A, Type.FLOAT)
        .fmt("GT")
        .fmt("DP", Number.ONE, Type.INTEGER)
        .record(
            "chr1",
            3,
            ref="A",
            alt=[Seq("G")],
            gt=["1|0", "0|0"],
            info={"AF": [0.25]},
            DP=[[10], [20]],
        )
        .record(
            "chr1",
            7,
            ref="C",
            alt=[Seq("CAT")],
            gt=["0|1", "1|1"],
            info={"AF": [0.75]},
            DP=[[5], [30]],
        )
    )
    vcf = doc.write(tmp_path / "fields.vcf.gz", bgzip=True, index=True)
    return vcf, ref


def test_reroute_false_carries_fields_matching_source(tmp_path):
    """reroute=False field carry-through: decoding the view with the same
    `with_fields([...])` selection must match decoding the SOURCE, for a
    full-coverage-identity view (all samples, full region) -- the simplest
    and strongest form of this check."""
    vcf, ref = _fields_vcf_and_ref(tmp_path)

    src_path = tmp_path / "src.svar2"
    SparseVar2.from_vcf(
        src_path,
        vcf,
        str(ref),
        info_fields=["AF"],
        format_fields=["DP"],
        threads=1,
    )
    src = SparseVar2(src_path)

    out = tmp_path / "out.svar2"
    src.write_view(
        ("chr1", 0, 40),
        src.available_samples,
        out,
        fields=["AF", "DP"],
        reroute=False,
        overwrite=True,
    )

    src_dec = src.with_fields(["AF", "DP"]).decode("chr1", [(0, 40)])
    out_dec = SparseVar2(out).with_fields(["AF", "DP"]).decode("chr1", [(0, 40)])

    assert out_dec.shape == src_dec.shape
    for key in ("pos", "ilen", "allele", "AF", "DP"):
        np.testing.assert_array_equal(
            np.asarray(src_dec[key].data), np.asarray(out_dec[key].data)
        )


def test_reroute_false_preserves_representation(svar2_store, tmp_path):
    """A source-dense variant stays dense under reroute=False (unlike
    reroute=True, which may re-route it under the subset's own cost model)."""
    sv = SparseVar2(svar2_store)
    contig = sv.contigs[0]
    out = tmp_path / "f.svar2"
    sv.write_view(
        (contig, 0, 40), sv.available_samples, out, reroute=False, overwrite=True
    )

    _ii, sd_src, *_ = _core.svar2_variant_stats(
        str(sv.path), contig, list(range(sv.n_samples))
    )
    _ii2, sd_out, *_ = _core.svar2_variant_stats(
        str(out), contig, list(range(sv.n_samples))
    )
    assert int(sd_src.sum()) > 0, "fixture must contain >=1 source-dense variant"
    assert int(sd_src.sum()) == int(sd_out.sum())


def test_reroute_false_recomputes_signatures(tmp_path):
    """reroute=False + reference= reproduces a from-scratch `mutcat`, honestly
    recomputed (not copied): for a full-coverage-identity view, the output's
    mutcat sidecar must be byte-identical to the source's own from-scratch
    annotation (both computed from the SAME reference sequence)."""
    ref = tmp_path / "ref.fa"
    ref.write_text(f">chr1\n{_REF_SEQ}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    doc = (
        VcfBuilder(samples=["S0", "S1"], contigs=[("chr1", 40)])
        .fmt("GT")
        .record("chr1", 3, ref="A", alt=[Seq("G")], gt=["1|0", "0|0"])
        .record("chr1", 20, ref="T", alt=[Seq("C")], gt=["0|1", "1|1"])
    )
    vcf = doc.write(tmp_path / "sig.vcf.gz", bgzip=True, index=True)

    src_path = tmp_path / "src.svar2"
    SparseVar2.from_vcf(src_path, vcf, str(ref), signatures=True, threads=1)
    src = SparseVar2(src_path)
    assert src._is_annotated(), "source must be annotated from_vcf(signatures=True)"

    out = tmp_path / "out.svar2"
    src.write_view(
        ("chr1", 0, 40),
        src.available_samples,
        out,
        reference=str(ref),
        reroute=False,
        overwrite=True,
    )
    viewed = SparseVar2(out)
    assert viewed._is_annotated()

    for rel in (
        "mutcat/var_key_snp/code.bin",
        "mutcat/var_key_snp/ref.bin",
        "mutcat/dense_snp/code.bin",
        "mutcat/dense_snp/ref.bin",
        "mutcat/var_key_indel/code.bin",
        "mutcat/dense_indel/code.bin",
    ):
        src_bytes = _read_if_exists(src_path / "chr1" / rel)
        out_bytes = _read_if_exists(out / "chr1" / rel)
        assert src_bytes == out_bytes, rel
    # Non-vacuity: the code sidecar actually has content (not a None==None pass).
    assert (out / "chr1" / "mutcat" / "var_key_snp" / "code.bin").stat().st_size > 0


def test_reroute_false_dtype_preserved(tmp_path):
    """A field with a narrow stored dtype (u16) keeps that dtype in the
    reroute=False output's meta.json (byte-copy preserves dtype -- there is
    no widen/narrow step on this path)."""
    doc = (
        VcfBuilder(samples=["S0", "S1"], contigs=[("chr1", 100_000)])
        .fmt("GT")
        .fmt("DP", Number.ONE, Type.INTEGER)
        .record(
            "chr1", 100, ref="A", alt=[Seq("C")], gt=["0|1", "1|1"], DP=[[10], [20]]
        )
    )
    vcf = doc.write(tmp_path / "u16.vcf.gz", bgzip=True, index=True)

    src_path = tmp_path / "src.svar2"
    SparseVar2.from_vcf(
        src_path,
        vcf,
        no_reference=True,
        format_fields=[FormatField("DP", dtype="u16")],
    )
    src = SparseVar2(src_path)
    src_meta = json.loads((src_path / "meta.json").read_text())
    assert next(f for f in src_meta["fields"] if f["name"] == "DP")["dtype"] == "u16"

    out = tmp_path / "out.svar2"
    src.write_view(
        ("chr1", 0, 100_000),
        src.available_samples,
        out,
        fields=["DP"],
        reroute=False,
        overwrite=True,
    )
    out_meta = json.loads((out / "meta.json").read_text())
    dp = next(f for f in out_meta["fields"] if f["name"] == "DP")
    assert dp["dtype"] == "u16"


def test_reroute_false_lut_indels_decode(tmp_path):
    """A store with a long-allele indel (ALT longer than the inline threshold,
    routed through the long-allele LUT), sliced with reroute=False, still
    decodes that indel's ALT correctly: the output's decode must match the
    SOURCE's decode exactly (verbatim LUT bytes + unchanged keys)."""
    insertion = "CGTA" * 7  # 28bp, well past svar2-codec's 13-byte inline cap
    doc = (
        VcfBuilder(samples=["S0", "S1"], contigs=[("chr1", 100_000)])
        .fmt("GT")
        .record(
            "chr1",
            100,
            ref="A",
            alt=[Seq("A" + insertion)],
            gt=["0|1", "0|0"],
        )
    )
    vcf = doc.write(tmp_path / "longindel.vcf.gz", bgzip=True, index=True)

    src_path = tmp_path / "src.svar2"
    SparseVar2.from_vcf(src_path, vcf, no_reference=True)
    src = SparseVar2(src_path)

    out = tmp_path / "out.svar2"
    src.write_view(
        ("chr1", 0, 100_000),
        src.available_samples,
        out,
        reroute=False,
        overwrite=True,
    )

    src_dec = src.decode("chr1", [(0, 100_000)])
    out_dec = SparseVar2(out).decode("chr1", [(0, 100_000)])

    src_ilen = np.asarray(src_dec["ilen"].data)
    assert (src_ilen > 13).any(), "fixture must actually exercise the long-allele LUT"

    np.testing.assert_array_equal(src_ilen, np.asarray(out_dec["ilen"].data))
    np.testing.assert_array_equal(
        np.asarray(src_dec["allele"].data), np.asarray(out_dec["allele"].data)
    )

"""End-to-end tests for ``SparseVar2.annotate_clusters``.

Fixture (3 samples, chr1): a doublet at 100/101 carried by s0+s1 (routed
DENSE because VAF is stored f16 -- see `_store`), rare SNPs at 200/5000 and a
rare indel at 300 carried by s2 (var_key / var_key_indel). With imd_cutoff=1000
the doublet is class 1, s2's lone SNPs are class 0, and the indel is 255.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vcfixture import Number, Seq, Type, VcfBuilder

from genoray import SparseVar2
from genoray._svar2_clusters import (
    CLUSTER_VERSION,
    DOUBLET,
    KATAEGIS,
    MBS,
    NONCLUSTERED,
    NOT_ANNOTATED,
    OMIKLI,
    OTHER,
)
from genoray._svar2_fields import FormatField


def _fixture(tmp_path: Path) -> Path:
    doc = (
        VcfBuilder(samples=["s0", "s1", "s2"], contigs=[("chr1", None)])
        .fmt("GT")
        .fmt("VAF", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="C",
            alt=[Seq("A")],
            gt=["1|0", "1|0", "0|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1",
            101,
            ref="C",
            alt=[Seq("G")],
            gt=["1|0", "0|1", "0|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1",
            200,
            ref="A",
            alt=[Seq("T")],
            gt=["0|0", "0|0", "1|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1",
            300,
            ref="A",
            alt=[Seq("AT")],
            gt=["0|0", "0|0", "0|1"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1",
            5000,
            ref="G",
            alt=[Seq("T")],
            gt=["0|0", "0|0", "1|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
    )
    return doc.write(tmp_path / "clusters.vcf.gz", bgzip=True, index=True)


def _store(tmp_path: Path) -> SparseVar2:
    # VAF is deliberately f16: the cost model bills a FORMAT field
    # `format_bits * n_samples` in dense but `format_bits * x_calls` in
    # var_key. With 3 samples and a 2-call doublet, an f32 field (32 bits)
    # costs dense 96 vs var_key 64 and routes everything var_key; at 16 bits
    # the doublet routes dense_snp while the rare SNPs stay var_key_snp and
    # the lone indel var_key_indel.
    out = tmp_path / "clusters.svar2"
    SparseVar2.from_vcf(
        out,
        _fixture(tmp_path),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f16")],
    )
    return SparseVar2(out)


def test_codebook_is_pinned():
    assert (NONCLUSTERED, DOUBLET, MBS, OMIKLI, KATAEGIS, OTHER, NOT_ANNOTATED) == (
        0,
        1,
        2,
        3,
        4,
        5,
        255,
    )
    assert CLUSTER_VERSION == 1


def test_annotate_clusters_end_to_end(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")

    meta = json.loads((store.path / "meta.json").read_text())
    assert meta["cluster_version"] == CLUSTER_VERSION
    assert meta["cluster_contigs"] == ["chr1"]
    assert meta["cluster_cutoff"] == 1000.0
    assert meta["cluster_vaf_field"] == "VAF"
    assert meta["cluster_vaf_cut"] == 0.1
    assert {
        "name": "cluster_class",
        "category": "format",
        "dtype": "u8",
        "default": None,
    } in meta["fields"]

    sv = SparseVar2(store.path).with_fields(["cluster_class"])
    rag = sv.decode("chr1", [(0, 1_000_000)])
    # carrier-only flat (R, S, P) order: s0h0, s0h1, s1h0, s1h1, s2h0, s2h1
    assert rag["pos"].lengths.reshape(-1).tolist() == [2, 0, 1, 1, 2, 1]
    labels = np.asarray(rag["cluster_class"].data)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                DOUBLET,
                DOUBLET,
                DOUBLET,
                DOUBLET,
                NONCLUSTERED,
                NONCLUSTERED,
                NOT_ANNOTATED,
            ],
            dtype=labels.dtype,
        ),
    )


def test_dense_stream_is_dense_row_major(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")

    dense_dir = store.path / "chr1" / "dense" / "snp"
    positions = np.frombuffer(
        (dense_dir / "positions.bin").read_bytes(), dtype=np.uint32
    )
    assert len(positions) >= 2, "fixture must route the recurrent doublet dense"
    values = np.frombuffer(
        (
            store.path
            / "chr1"
            / "fields"
            / "format"
            / "cluster_class"
            / "dense_snp"
            / "values.bin"
        ).read_bytes(),
        dtype=np.uint8,
    )
    assert len(values) == len(positions) * 3
    # Dense stream row-major (variant-major, sample-minor): the fixture's only
    # dense variants are the doublet, carried by samples 0 and 1; sample 2 is
    # not a carrier and stays 255.
    expected = np.full(len(positions) * 3, NOT_ANNOTATED, dtype=np.uint8)
    for row in range(len(positions)):
        expected[row * 3 + 0] = DOUBLET
        expected[row * 3 + 1] = DOUBLET
    np.testing.assert_array_equal(values, expected)


def test_out_of_scope_contigs_are_255_filled(tmp_path: Path):
    # second contig, one lone SNP, untouched by contigs=["chr1"]
    doc = (
        VcfBuilder(samples=["s0", "s1", "s2"], contigs=[("chr1", None), ("chr2", None)])
        .fmt("GT")
        .fmt("VAF", Number.ONE, Type.FLOAT)
        .record(
            "chr1",
            100,
            ref="C",
            alt=[Seq("A")],
            gt=["1|0", "1|0", "0|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr1",
            101,
            ref="C",
            alt=[Seq("G")],
            gt=["1|0", "0|1", "0|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
        .record(
            "chr2",
            50,
            ref="A",
            alt=[Seq("T")],
            gt=["1|0", "0|0", "0|0"],
            VAF=[[0.5], [0.5], [0.5]],
        )
    )
    out = tmp_path / "two.svar2"
    SparseVar2.from_vcf(
        out,
        doc.write(tmp_path / "two.vcf.gz", bgzip=True, index=True),
        no_reference=True,
        format_fields=[FormatField("VAF", dtype="f32")],
    )
    store = SparseVar2(out)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF", contigs=["chr1"])

    meta = json.loads((out / "meta.json").read_text())
    assert meta["cluster_contigs"] == ["chr1"]
    chr2_values = out / "chr2" / "fields" / "format" / "cluster_class"
    assert (chr2_values / "var_key_snp" / "values.bin").read_bytes() == bytes(
        [NOT_ANNOTATED]
    )
    assert (chr2_values / "var_key_indel" / "values.bin").stat().st_size == 0
    sv = SparseVar2(out).with_fields(["cluster_class"])
    rag = sv.decode("chr2", [(0, 1_000_000)])
    assert np.asarray(rag["cluster_class"].data).tolist() == [NOT_ANNOTATED]


def test_validation_errors(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="unknown samples"):
        store.annotate_clusters(imd_cutoff={"s0": 10.0, "nope": 10.0})
    with pytest.raises(ValueError, match="missing samples"):
        store.annotate_clusters(imd_cutoff={"s0": 10.0})
    with pytest.raises(ValueError, match="must be > 0"):
        store.annotate_clusters(imd_cutoff=0.0)
    with pytest.raises(ValueError, match="vaf_cut"):
        store.annotate_clusters(imd_cutoff=10.0, vaf_field="VAF", vaf_cut=0.0)
    with pytest.raises(ValueError, match="not in the store"):
        store.annotate_clusters(imd_cutoff=10.0, vaf_field="NOPE")
    with pytest.raises(ValueError, match="not found in store"):
        store.annotate_clusters(imd_cutoff=10.0, contigs=["chr9"])


def test_failed_write_leaves_meta_unstamped(tmp_path: Path):
    store = _store(tmp_path)
    before = (store.path / "meta.json").read_bytes()
    # Block the var_key_snp staging path with a directory so the Rust write
    # fails before anything is renamed into place.
    blocked = (
        store.path / "chr1" / "fields" / "format" / "cluster_class" / "var_key_snp"
    )
    blocked.mkdir(parents=True, exist_ok=True)
    (blocked / "values.bin.tmp").mkdir(exist_ok=True)
    with pytest.raises(OSError):
        store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    assert (store.path / "meta.json").read_bytes() == before
    assert not (blocked / "values.bin").exists()


def test_rerun_is_idempotent(tmp_path: Path):
    store = _store(tmp_path)
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    first = (
        store.path
        / "chr1"
        / "fields"
        / "format"
        / "cluster_class"
        / "dense_snp"
        / "values.bin"
    ).read_bytes()
    store.annotate_clusters(imd_cutoff=1000.0, vaf_field="VAF")
    second = (
        store.path
        / "chr1"
        / "fields"
        / "format"
        / "cluster_class"
        / "dense_snp"
        / "values.bin"
    ).read_bytes()
    assert first == second

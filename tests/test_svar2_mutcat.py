from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from genoray import SparseVar2
from tests.test_svar2_from_vcf import _write_ref, _write_vcf


def test_mutation_matrix_shape_and_labels(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store.svar2"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)

    sv2 = SparseVar2(out)
    sv2.annotate_mutations(ref)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["mutcat_contigs"] == ["chr1"]

    mm = sv2.mutation_matrix("SBS96", count="allele")
    assert mm.columns[0] == "MutationType"
    assert mm.height == 96
    assert set(mm.columns[1:]) == set(sv2.available_samples)
    assert mm.select(pl.exclude("MutationType")).to_numpy().dtype.kind in "iu"


def test_mutation_matrix_requires_annotation(tmp_path: Path):
    ref = _write_ref(tmp_path)
    vcf = _write_vcf(tmp_path, symbolic=False, indexed=True)
    out = tmp_path / "store_unannotated.svar2"
    SparseVar2.from_vcf(out, vcf, ref, threads=1)

    sv2 = SparseVar2(out)
    with pytest.raises(ValueError, match="not annotated"):
        sv2.mutation_matrix("SBS96")

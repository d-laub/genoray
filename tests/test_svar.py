from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest import fixture
from pytest_cases import parametrize_with_cases
from seqpro.rag import OFFSET_TYPE, lengths_to_offsets

from genoray import SparseVar
from genoray._svar import DOSAGE_TYPE, V_IDX_TYPE
from seqpro.rag import Ragged

ddir = Path(__file__).parent / "data"

N_SAMPLES = 2
PLOIDY = 2
DATA = np.array([2, 5, 0, 4, 0, 3, 0, 1, 3, 4], V_IDX_TYPE)
DOSAGES = np.array([0.9, 0.9, 1, 1, 2, 2, 2, 1, 2, 1], DOSAGE_TYPE)
LENGTHS = np.array([[2, 2], [2, 4]])
OFFSETS = lengths_to_offsets(LENGTHS)
_, counts = np.unique(DATA, return_counts=True)
afs = counts / (N_SAMPLES * PLOIDY)


def get_missing_contig_desired(
    svar: SparseVar, n_ranges: int, n_samples: int
) -> Ragged[V_IDX_TYPE]:
    # (r s p 2)
    offsets = np.full((2, n_ranges, n_samples, svar.ploidy), -1, OFFSET_TYPE)
    return Ragged[V_IDX_TYPE].from_offsets(
        svar.genos.data, (n_ranges, n_samples, svar.ploidy), offsets.reshape(2, -1)
    )


def svar_vcf():
    svar = SparseVar(ddir / "biallelic.vcf.svar", "AF", fields=["dosages"])
    return svar


def svar_pgen():
    svar = SparseVar(ddir / "biallelic.pgen.svar", "AF", fields=["dosages"])
    return svar


@fixture
def svar():
    svar = SparseVar(ddir / "biallelic.vcf.svar", "AF")
    return svar


@parametrize_with_cases("svar", cases=".", prefix="svar_")
def test_contents(svar: SparseVar):
    # (s p)
    lengths = LENGTHS
    desired_genos = Ragged[V_IDX_TYPE].from_lengths(DATA, lengths)
    desired_dosages = Ragged[DOSAGE_TYPE].from_lengths(DOSAGES, lengths)

    if svar.path.suffixes[0] == ".vcf":
        assert svar.contigs == ["chr1", "chr2", "chr3"]
    elif svar.path.suffixes[0] == ".pgen":
        assert svar.contigs == ["1", "2"]

    assert svar.genos.shape == desired_genos.shape
    np.testing.assert_equal(svar.genos.data, desired_genos.data)
    np.testing.assert_equal(svar.genos.offsets, desired_genos.offsets)

    dosages = svar.fields.get("dosages")
    assert dosages is not None
    assert dosages.shape == desired_genos.shape
    np.testing.assert_allclose(dosages.data, desired_dosages.data, atol=5e-5)
    np.testing.assert_equal(dosages.offsets, desired_dosages.offsets)


def case_all():
    cse = "chr1", 81261, 81265
    # (r 2)
    var_ranges = np.array([[0, 3]], V_IDX_TYPE)
    # (s p)
    shape = (1, N_SAMPLES, PLOIDY, None)
    offsets = np.array([[0, 2, 4, 6], [1, 3, 5, 8]], dtype=OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


def case_spanning_del():
    cse = "chr1", 81262, 81263
    # (r 2)
    var_ranges = np.array([[0, 1]], V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY, None)
    # (s p)
    offsets = np.array([[0, 2, 4, 6], [0, 3, 5, 7]], dtype=OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


def case_missing_contig():
    cse = "🥸", 81261, 81263
    # (r 2)
    var_ranges = np.full((1, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY, None)
    # (r s p 2)
    offsets = np.full((2, N_SAMPLES, PLOIDY, 1), -1, OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets.reshape(2, -1))
    return cse, var_ranges, desired


def case_no_vars():
    cse = "chr1", int(1e8), int(2e8)
    # (r 2)
    var_ranges = np.full((1, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY, None)
    # (2 r s p)
    offsets = np.full((2, N_SAMPLES, PLOIDY, 1), np.iinfo(OFFSET_TYPE).max, OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets.reshape(2, -1))
    return cse, var_ranges, desired


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_var_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: Ragged[V_IDX_TYPE] | None,
):
    actual = svar.var_ranges(*cse)

    np.testing.assert_equal(actual, var_ranges)


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_read_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: Ragged[V_IDX_TYPE] | None,
):
    actual = svar.read_ranges(*cse)

    if desired is None:
        desired = get_missing_contig_desired(svar, 1, svar.n_samples)

    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_read_ranges_sample_subset(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: Ragged[V_IDX_TYPE] | None,
):
    sample = "sample2"
    s_idx = svar.available_samples.index(sample)
    actual = svar.read_ranges(*cse, samples=sample)

    if desired is None:
        desired = get_missing_contig_desired(svar, 1, svar.n_samples)

    # desired: (1 s p ~v)
    desired = desired[:, [s_idx]]
    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


def test_read_ranges_sample_reorder(svar: SparseVar):
    cse = "chr1", 81261, 81265
    actual = svar.read_ranges(*cse, samples=["sample2", "sample1"])
    desired = svar.read_ranges(*cse)[:, [1, 0]]

    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


def length_no_ext():
    cse = "chr1", 81264, 81265
    shape = (1, 2, 2, None)
    # (s p)
    offsets = np.array([[0, 3, 5, 8], [1, 3, 5, 8]], OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets)
    return cse, desired


def length_ext():
    cse = "chr1", 81262, 81263
    shape = (1, N_SAMPLES, PLOIDY, None)
    # (s p)
    offsets = np.array([[0, 2, 4, 6], [0, 3, 5, 8]], OFFSET_TYPE)
    desired = Ragged[V_IDX_TYPE].from_offsets(DATA, shape, offsets)
    return cse, desired


@parametrize_with_cases("cse, desired", cases=".", prefix="length_")
def test_read_ranges_with_length(
    svar: SparseVar, cse: tuple[str, int, int], desired: Ragged[V_IDX_TYPE]
):
    actual = svar.read_ranges_with_length(*cse)

    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


def test_attrs_numeric_in_index():
    svar = SparseVar(ddir / "biallelic.vcf.svar", attrs="AF")
    assert "AF" in svar.index.columns
    assert svar.index["AF"].dtype.is_numeric()
    np.testing.assert_allclose(svar.index["AF"].to_numpy(), afs, atol=1e-6)


def test_attrs_non_numeric_raises():
    with pytest.raises(ValueError, match="numeric"):
        SparseVar(ddir / "biallelic.vcf.svar", attrs="REF")


def test_attrs_not_in_index_without_request():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    assert "AF" not in svar.index.columns


def test_compute_afs(svar: SparseVar):
    actual_afs = svar._compute_afs()
    np.testing.assert_equal(actual_afs, afs)


def test_cache_afs(svar: SparseVar):
    np.testing.assert_equal(svar.index["AF"], afs)


def test_with_fields_adds_fields(svar: SparseVar):
    svar_with = svar.with_fields(["dosages"])
    plain = svar.read_ranges("chr1", 81261, 81265)
    result = svar_with.read_ranges("chr1", 81261, 81265)
    # record exposes .genos and .dosages with matching outer shape
    assert result.genos.shape == plain.shape
    assert result.dosages.shape == plain.shape
    assert result.dosages.data.dtype == np.float32
    # genos data values are a subset of the full genos memmap
    assert set(result.genos.data.tolist()).issubset(set(svar.genos.data.tolist()))


def test_with_fields_none_is_identity(svar: SparseVar):
    shallow = svar.with_fields(None)
    assert shallow.fields == svar.fields
    assert shallow.genos is svar.genos


def test_with_fields_false_drops_fields():
    svar_with = SparseVar(ddir / "biallelic.vcf.svar", fields=["dosages"])
    svar_no_fields = svar_with.with_fields(False)
    assert svar_no_fields.fields == {}
    plain = svar_no_fields.read_ranges("chr1", 81261, 81265)
    with_genos = svar_with.read_ranges("chr1", 81261, 81265).genos
    # both should have the same outer shape
    assert plain.shape == with_genos.shape


def test_fields_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        SparseVar(ddir / "biallelic.vcf.svar", fields=["nonexistent"])


def test_with_fields_missing_raises():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    with pytest.raises(ValueError, match="not found"):
        svar.with_fields(["nonexistent"])


def test_available_fields(svar: SparseVar):
    assert "dosages" in svar.available_fields
    assert svar.available_fields["dosages"] == np.dtype(DOSAGE_TYPE)


def test_svar_nbytes_index_only():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    # nbytes counts only the resident polars index, not the mmap'd genos/fields
    assert svar.nbytes == svar.index.estimated_size()
    assert svar.nbytes > 0

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest import fixture
from pytest_cases import parametrize_with_cases

from genoray import SparseVar
from genoray._svar import V_IDX_TYPE, SparseGenotypes

ddir = Path(__file__).parent / "data"

DATA = np.array([2, 5, 0, 4, 0, 3, 0, 1, 3, 4], V_IDX_TYPE)


def svar_vcf():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    return svar


def svar_pgen():
    svar = SparseVar(ddir / "biallelic.pgen.svar")
    return svar


@fixture
def svar():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    return svar


@parametrize_with_cases("svar", cases=".", prefix="svar_")
def test_contents(svar: SparseVar):
    # (s p)
    lengths = np.array([[2, 2], [2, 4]], np.uint32)
    desired = SparseGenotypes.from_lengths(DATA, lengths)

    if svar.path.suffixes[0] == ".vcf":
        assert svar.contigs == ["chr1", "chr2"]
    elif svar.path.suffixes[0] == ".pgen":
        assert svar.contigs == ["1", "2"]
    assert svar.genos.shape == desired.shape
    np.testing.assert_equal(svar.genos.data, desired.data)
    np.testing.assert_equal(svar.genos.offsets, desired.offsets)


def case_all():
    cse = "chr1", 81261, 81265
    # (r 2)
    var_ranges = np.array([[0, 3]], V_IDX_TYPE)
    # (s p)
    shape = (1, 2, 2)
    offsets = np.array([[0, 1], [2, 3], [4, 5], [6, 8]])
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


def case_spanning_del():
    cse = "chr1", 81262, 81263
    # (r 2)
    var_ranges = np.array([[0, 1]], V_IDX_TYPE)
    shape = (1, 2, 2)
    # (s p)
    offsets = np.array([[0, 0], [2, 3], [4, 5], [6, 7]])
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_var_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: SparseGenotypes,
):
    actual = svar.var_ranges(*cse)

    np.testing.assert_equal(actual, var_ranges)


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_read_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: SparseGenotypes,
):
    actual = svar.read_ranges(*cse)

    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_read_ranges_sample_subset(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: SparseGenotypes,
):
    samples = "sample2"
    actual = svar.read_ranges(*cse, samples=samples)

    # desired: (1 s p ~v)
    desired = SparseGenotypes.from_awkward(desired.to_awkward()[:, [1]])  # type: ignore
    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


def length_no_ext():
    cse = "chr1", 81264, 81265
    shape = (1, 2, 2)
    # (s p)
    offsets = np.array([[0, 1], [3, 3], [5, 5], [8, 8]])
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, desired


def length_ext():
    cse = "chr1", 81262, 81263
    shape = (1, 2, 2)
    # (s p)
    offsets = np.array([[0, 1], [2, 3], [4, 5], [6, 8]])
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, desired


@parametrize_with_cases("cse, desired", cases=".", prefix="length_")
def test_read_ranges_with_length(
    svar: SparseVar, cse: tuple[str, int, int], desired: SparseGenotypes
):
    actual = svar.read_ranges_with_length(*cse)

    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)

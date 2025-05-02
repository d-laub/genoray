from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest import fixture
from pytest_cases import parametrize_with_cases
from seqpro._ragged import OFFSET_TYPE

from genoray import SparseVar
from genoray._svar import (
    CCF_TYPE,
    POS_TYPE,
    V_IDX_TYPE,
    SparseCCFs,
    SparseGenotypes,
    _infer_germline_ccfs,
)

ddir = Path(__file__).parent / "data"

DATA = np.array([2, 5, 0, 4, 0, 3, 0, 1, 3, 4], V_IDX_TYPE)
CCFS = np.array([0.9, 0.9, 1, 1, 2, 2, 2, 1, 2, 1], CCF_TYPE)


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
    desired_genos = SparseGenotypes.from_lengths(DATA, lengths)
    desired_ccfs = SparseCCFs.from_lengths(CCFS, lengths)

    if svar.path.suffixes[0] == ".vcf":
        assert svar.contigs == ["chr1", "chr2"]
    elif svar.path.suffixes[0] == ".pgen":
        assert svar.contigs == ["1", "2"]

    assert svar.genos.shape == desired_genos.shape
    np.testing.assert_equal(svar.genos.data, desired_genos.data)
    np.testing.assert_equal(svar.genos.offsets, desired_genos.offsets)

    assert svar.ccfs is not None
    assert svar.ccfs.shape == desired_genos.shape
    np.testing.assert_allclose(svar.ccfs.data, desired_ccfs.data, atol=5e-5)
    np.testing.assert_equal(svar.ccfs.offsets, desired_ccfs.offsets)


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


def ccfs_no_overlaps():
    ccfs = np.array([np.nan, 0.1, 0.2, 0.3], np.float32)
    v_idxs = np.array([0, 1, 2, 3], V_IDX_TYPE)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    np.nan_to_num(desired, False, max_ccf)

    return ccfs, v_idxs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_no_germs():
    ccfs = np.array([0.1, 0.1, 0.2, 0.3], np.float32)
    v_idxs = np.array([0, 1, 2, 3], V_IDX_TYPE)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()

    return ccfs, v_idxs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_overlap():
    ccfs = np.array([0.1, np.nan, 0.2, 0.3], np.float32)
    v_idxs = np.array([0, 1, 2, 3], V_IDX_TYPE)
    v_starts = np.array([0, 1, 1, 1], POS_TYPE)
    ilens = np.array([0, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    desired[1] = max_ccf - ccfs[2:].sum()

    return ccfs, v_idxs, v_starts, ilens, v_offsets, max_ccf, desired


def ccfs_spanning_del():
    ccfs = np.array([0.1, np.nan, 0.2, 0.3], np.float32)
    v_idxs = np.array([0, 1, 2, 3], V_IDX_TYPE)
    v_starts = np.array([0, 1, 2, 3], POS_TYPE)
    ilens = np.array([-1, 0, 0, 0], np.int32)
    v_offsets = np.array([0, 4], OFFSET_TYPE)
    max_ccf = 1.0

    desired = ccfs.copy()
    desired[1] = max_ccf - ccfs[0]

    return ccfs, v_idxs, v_starts, ilens, v_offsets, max_ccf, desired


@parametrize_with_cases(
    "ccfs, v_idxs, v_starts, ilens, v_offsets, max_ccf, desired",
    cases=".",
    prefix="ccfs_",
)
def test_infer_germ_ccfs(
    ccfs: NDArray[np.float32],
    v_idxs: NDArray[V_IDX_TYPE],
    v_starts: NDArray[POS_TYPE],
    ilens: NDArray[np.int32],
    v_offsets: NDArray[OFFSET_TYPE],
    max_ccf: float,
    desired: NDArray[np.float32],
):
    _infer_germline_ccfs(
        ccfs=ccfs,
        v_idxs=v_idxs,
        v_starts=v_starts,
        ilens=ilens,
        v_offsets=v_offsets,
        max_ccf=max_ccf,
    )
    np.testing.assert_equal(ccfs, desired)

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest import fixture
from pytest_cases import parametrize_with_cases
from seqpro._ragged import LENGTH_TYPE, OFFSET_TYPE, lengths_to_offsets

from genoray import SparseVar
from genoray._svar import DOSAGE_TYPE, V_IDX_TYPE, SparseDosages, SparseGenotypes

ddir = Path(__file__).parent / "data"

N_SAMPLES = 2
PLOIDY = 2
DATA = np.array([2, 5, 0, 4, 0, 3, 0, 1, 3, 4], V_IDX_TYPE)
DOSAGES = np.array([0.9, 0.9, 1, 1, 2, 2, 2, 1, 2, 1], DOSAGE_TYPE)
LENGTHS = np.array([[2, 2], [2, 4]], LENGTH_TYPE)
OFFSETS = lengths_to_offsets(LENGTHS)


def get_missing_contig_desired(
    svar: SparseVar, n_ranges: int, n_samples: int
) -> SparseGenotypes:
    # (r s p 2)
    offsets = np.full((2, n_ranges, n_samples, svar.ploidy), -1, OFFSET_TYPE)
    return SparseGenotypes.from_offsets(
        svar.genos.data, (n_ranges, n_samples, svar.ploidy), offsets.reshape(2, -1)
    )


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
    lengths = LENGTHS
    desired_genos = SparseGenotypes.from_lengths(DATA, lengths)
    desired_ccfs = SparseDosages.from_lengths(DOSAGES, lengths)

    if svar.path.suffixes[0] == ".vcf":
        assert svar.contigs == ["chr1", "chr2"]
    elif svar.path.suffixes[0] == ".pgen":
        assert svar.contigs == ["1", "2"]

    assert svar.genos.shape == desired_genos.shape
    np.testing.assert_equal(svar.genos.data, desired_genos.data)
    np.testing.assert_equal(svar.genos.offsets, desired_genos.offsets)

    assert svar.dosages is not None
    assert svar.dosages.shape == desired_genos.shape
    np.testing.assert_allclose(svar.dosages.data, desired_ccfs.data, atol=5e-5)
    np.testing.assert_equal(svar.dosages.offsets, desired_ccfs.offsets)


def case_all():
    cse = "chr1", 81261, 81265
    # (r 2)
    var_ranges = np.array([[0, 3]], V_IDX_TYPE)
    # (s p)
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.array([[0, 2, 4, 6], [1, 3, 5, 8]], dtype=OFFSET_TYPE)
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


def case_spanning_del():
    cse = "chr1", 81262, 81263
    # (r 2)
    var_ranges = np.array([[0, 1]], V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.array([[0, 2, 4, 6], [0, 3, 5, 7]], dtype=OFFSET_TYPE)
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, var_ranges, desired


def case_missing_contig():
    cse = "🥸", 81261, 81263
    # (r 2)
    var_ranges = np.full((1, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.full((2, *shape), -1, OFFSET_TYPE)
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets.reshape(2, -1))
    return cse, var_ranges, desired


def case_no_vars():
    cse = "chr1", int(1e8), int(2e8)
    # (r 2)
    var_ranges = np.full((1, 2), np.iinfo(V_IDX_TYPE).max, V_IDX_TYPE)
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.full((2, *shape), np.iinfo(OFFSET_TYPE).max, OFFSET_TYPE)
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets.reshape(2, -1))
    return cse, var_ranges, desired


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_var_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: SparseGenotypes | None,
):
    actual = svar.var_ranges(*cse)

    np.testing.assert_equal(actual, var_ranges)


@parametrize_with_cases("cse, var_ranges, desired", cases=".", prefix="case_")
def test_read_ranges(
    svar: SparseVar,
    cse: tuple[str, int, int],
    var_ranges: NDArray[V_IDX_TYPE],
    desired: SparseGenotypes | None,
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
    desired: SparseGenotypes | None,
):
    sample = "sample2"
    s_idx = svar.available_samples.index(sample)
    actual = svar.read_ranges(*cse, samples=sample)

    if desired is None:
        desired = get_missing_contig_desired(svar, 1, svar.n_samples)

    # desired: (1 s p ~v)
    desired = SparseGenotypes.from_awkward(desired.to_awkward()[:, [s_idx]])  # type: ignore
    assert actual.shape == desired.shape
    np.testing.assert_equal(actual.data, desired.data)
    np.testing.assert_equal(actual.offsets, desired.offsets)


def length_no_ext():
    cse = "chr1", 81264, 81265
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.array([[0, 3, 5, 8], [1, 3, 5, 8]], dtype=OFFSET_TYPE)
    desired = SparseGenotypes.from_offsets(DATA, shape, offsets)
    return cse, desired


def length_ext():
    cse = "chr1", 81262, 81263
    shape = (1, N_SAMPLES, PLOIDY)
    # (2 r*s*p)
    offsets = np.array([[0, 2, 4, 6], [1, 3, 5, 8]], dtype=OFFSET_TYPE)
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

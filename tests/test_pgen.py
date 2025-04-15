from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest_cases import fixture, parametrize_with_cases

from genoray import PGEN

tdir = Path(__file__).parent
ddir = tdir / "data"


@fixture  # type: ignore
def pgen():
    return PGEN(ddir / "test.pgen")


def read_all():
    cse = "chr1", 81261, 81262
    # (s p v)
    genos = np.array([[[0, -1], [1, -1]], [[1, 0], [1, 1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0], [1, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan], [2.0, 1.0]], np.float32)
    return cse, genos, phasing, dosages


def read_spanning_del():
    cse = "chr1", 81262, 81263
    # (s p v)
    genos = np.array([[[0], [1]], [[1], [1]]], np.int8)
    # (s v)
    phasing = np.array([[1], [1]], np.bool_)
    dosages = np.array([[1.0], [2.0]], np.float32)
    return cse, genos, phasing, dosages


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_read(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    # (s p v)
    g = pgen.read(*cse)
    assert g is not None
    np.testing.assert_equal(g, genos)

    d = pgen.read(*cse, PGEN.Dosages)
    assert d is not None
    np.testing.assert_equal(d, dosages)

    gp = pgen.read(*cse, PGEN.GenosPhasing)
    assert gp is not None
    np.testing.assert_equal(gp[0], genos)
    np.testing.assert_equal(gp[1], phasing)

    gd = pgen.read(*cse, PGEN.GenosDosages)
    assert gd is not None
    np.testing.assert_equal(gd[0], genos)
    np.testing.assert_equal(gd[1], dosages)

    gpd = pgen.read(*cse, PGEN.GenosPhasingDosages)
    assert gpd is not None
    np.testing.assert_equal(gpd[0], genos)
    np.testing.assert_equal(gpd[1], phasing)
    np.testing.assert_equal(gpd[2], dosages)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    max_mem = pgen._mem_per_variant(PGEN.Genos)
    cat = partial(np.concatenate, axis=-1)
    itr = pgen.chunk(*cse, max_mem)
    chunks = list(itr)
    assert len(chunks) == genos.shape[-1]
    # assert len(chunks[1]) == dosages.shape[-1]
    actual = cat(chunks)
    np.testing.assert_equal(actual, genos)
    # np.testing.assert_equal(actual[1], dosages)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    s_idx = np.array([1], np.uint32)
    samples = [pgen.available_samples[i] for i in s_idx]
    pgen.set_samples(samples)
    assert pgen.current_samples == samples
    assert pgen.n_samples == len(samples)
    np.testing.assert_equal(pgen._s_idx, s_idx)

    # (s p v)
    actual_genos = pgen.read(*cse)
    assert actual_genos is not None
    actual_dosages = pgen.read(*cse, PGEN.Dosages)
    assert actual_dosages is not None
    actual_genos_dosages = pgen.read(*cse, PGEN.GenosDosages)
    assert actual_genos_dosages is not None
    np.testing.assert_equal(actual_genos, genos[s_idx])
    np.testing.assert_equal(actual_dosages, dosages[s_idx])
    np.testing.assert_equal(actual_genos_dosages[0], genos[s_idx])
    np.testing.assert_equal(actual_genos_dosages[1], dosages[s_idx])

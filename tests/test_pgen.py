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
    dosages = np.full((2, 2), 0.5, np.float32)
    dosages[0, 1] = np.nan
    return cse, genos, dosages


def read_spanning_del():
    cse = "chr1", 81262, 81263
    # (s p v)
    genos = np.array([[[0], [1]], [[1], [1]]], np.int8)
    # (s v)
    dosages = np.full((2, 1), 0.5, np.float32)
    return cse, genos, dosages


@parametrize_with_cases("cse, genos, dosages", cases=".", prefix="read_")
def test_read(
    pgen: PGEN[PGEN.Genos],
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    dosages: NDArray[np.float32],
):
    # (s p v)
    actual = pgen.read(*cse)
    np.testing.assert_equal(actual, genos)
    # np.testing.assert_equal(actual[1], dosages)


@parametrize_with_cases("cse, genos, dosages", cases=".", prefix="read_")
def test_read_chunks(
    pgen: PGEN[PGEN.Genos],
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    dosages: NDArray[np.float32],
):
    max_mem = pgen._mem_per_variant()
    cat = partial(np.concatenate, axis=-1)
    itr = pgen.read_chunks(*cse, max_mem)
    chunks = list(itr)
    assert len(chunks) == genos.shape[-1]
    # assert len(chunks[1]) == dosages.shape[-1]
    actual = cat(chunks)
    np.testing.assert_equal(actual, genos)
    # np.testing.assert_equal(actual[1], dosages)

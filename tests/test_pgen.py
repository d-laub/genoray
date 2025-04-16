from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest_cases import fixture, parametrize_with_cases

from genoray import PGEN

tdir = Path(__file__).parent
ddir = tdir / "data"


@fixture  # type: ignore
def pgen():
    return PGEN(ddir / "biallelic.pgen")


def read_all():
    cse = "chr1", 81261, 81262  # just 81262 in VCF
    # (s p v)
    genos = np.array([[[0, -1], [1, -1]], [[1, 0], [1, 1]]], np.int32)
    # (s v)
    phasing = np.array([[1, 0], [1, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan], [2.0, 1.0]], np.float32)
    return cse, genos, phasing, dosages


def read_spanning_del():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0], [1]], [[1], [1]]], np.int32)
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
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    gp = pgen.read(*cse, PGEN.GenosPhasing)
    assert gp is not None
    g, p = gp
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)

    gd = pgen.read(*cse, PGEN.GenosDosages)
    assert gd is not None
    g, d = gd
    np.testing.assert_equal(g, genos)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    gpd = pgen.read(*cse, PGEN.GenosPhasingDosages)
    assert gpd is not None
    g, p, d = gpd
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    mode = PGEN.GenosPhasingDosages
    gpd = pgen.chunk(*cse, pgen._mem_per_variant(mode), mode)
    for i, chunk in enumerate(gpd):
        g, p, d = chunk
        np.testing.assert_equal(g, genos[..., [i]])
        np.testing.assert_equal(p, phasing[..., [i]])
        np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_read_ranges(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    c, s, e = cse
    s = [s, s]
    e = [e, e]

    gpdo = pgen.read_ranges(c, s, e, PGEN.GenosPhasingDosages)
    assert gpdo is not None
    (g, p, d), o = gpdo
    np.testing.assert_equal(g[..., o[0] : o[1]], genos)
    np.testing.assert_equal(g[..., o[1] : o[2]], genos)
    np.testing.assert_equal(p[..., o[0] : o[1]], phasing)
    np.testing.assert_equal(p[..., o[1] : o[2]], phasing)
    np.testing.assert_allclose(d[..., o[0] : o[1]], dosages, rtol=1e-5)
    np.testing.assert_allclose(d[..., o[1] : o[2]], dosages, rtol=1e-5)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk_ranges(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    c, s, e = cse
    s = [s, s]
    e = [e, e]

    mode = PGEN.GenosPhasingDosages
    gpdo = pgen.chunk_ranges(c, s, e, max_mem=pgen._mem_per_variant(mode), mode=mode)
    for range_ in gpdo:
        assert range_ is not None
        for i, chunk in enumerate(range_):
            g, p, d = chunk
            np.testing.assert_equal(g, genos[..., [i]])
            np.testing.assert_equal(p, phasing[..., [i]])
            np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)


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

    gpd = pgen.read(*cse, PGEN.GenosPhasingDosages)
    assert gpd is not None
    g, p, d = gpd
    np.testing.assert_equal(g, genos[s_idx])
    np.testing.assert_equal(p, phasing[s_idx])
    np.testing.assert_allclose(d, dosages[s_idx], rtol=1e-5)


def length_no_ext():
    cse = "chr1", 81264, 81265  # just 81265 in VCF
    # (s p v)
    genos = np.array([[[1], [0]], [[-1], [-1]]], np.int8)
    # (s v)
    phasing = np.array([[1], [0]], np.bool_)
    dosages = np.array([[0.900024], [np.nan]], np.float32)
    return cse, genos, phasing, dosages


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0, -1, 1], [1, -1, 0]], [[1, 0, -1], [1, 1, -1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0, 1], [1, 0, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan, 0.900024], [2.0, 1.0, np.nan]], np.float32)
    return cse, genos, phasing, dosages


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="length_")
def test_chunk_with_length(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    mode = PGEN.GenosPhasingDosages
    max_mem = pgen._mem_per_variant(mode)
    gpd = pgen.chunk_ranges_with_length(*cse, max_mem, mode)
    for range_ in gpd:
        assert range_ is not None
        for chunk in range_:
            g, p, d = chunk
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_allclose(d, dosages, rtol=1e-5)

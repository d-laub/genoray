from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
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


def read_missing_contig():
    cse = "🥸", 81261, 81263
    # (s p v)
    genos = None
    # (s v)
    phasing = None
    dosages = None
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
    if cse[0] == "🥸":
        assert g is None
    else:
        assert g is not None
        np.testing.assert_equal(g, genos)

    d = pgen.read(*cse, PGEN.Dosages)
    if cse[0] == "🥸":
        assert g is None
    else:
        assert d is not None
        np.testing.assert_allclose(d, dosages, rtol=1e-5)

    gp = pgen.read(*cse, PGEN.GenosPhasing)
    if cse[0] == "🥸":
        assert g is None
    else:
        assert gp is not None
        g, p = gp
        np.testing.assert_equal(g, genos)
        np.testing.assert_equal(p, phasing)

    gd = pgen.read(*cse, PGEN.GenosDosages)
    if cse[0] == "🥸":
        assert g is None
    else:
        assert gd is not None
        g, d = gd
        np.testing.assert_equal(g, genos)
        np.testing.assert_allclose(d, dosages, rtol=1e-5)

    gpd = pgen.read(*cse, PGEN.GenosPhasingDosages)
    if cse[0] == "🥸":
        assert g is None
    else:
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
    chunk = None
    for i, chunk in enumerate(gpd):
        g, p, d = chunk
        np.testing.assert_equal(g, genos[..., [i]])
        np.testing.assert_equal(p, phasing[..., [i]])
        np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)
    if cse[0] == "🥸":
        assert chunk is None


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
    if cse[0] == "🥸":
        assert gpdo is None
    else:
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
        if cse[0] == "🥸":
            assert range_ is None
        else:
            assert range_ is not None
            for i, chunk in enumerate(range_):
                g, p, d = chunk
                np.testing.assert_equal(g, genos[..., [i]])
                np.testing.assert_equal(p, phasing[..., [i]])
                np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)


def samples_none():
    samples = None
    return samples


def samples_second():
    samples = "sample1"
    return samples


@parametrize_with_cases("samples", cases=".", prefix="samples_")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    samples: ArrayLike | None,
):
    pgen.set_samples(samples)

    if samples is None:
        samples = pgen.available_samples
        s_idx = slice(None)
    else:
        samples = np.atleast_1d(samples)
        s_idx = np.intersect1d(pgen.available_samples, samples, return_indices=True)[1]

    assert pgen.current_samples == samples
    assert pgen.n_samples == len(samples)
    np.testing.assert_equal(pgen._s_idx, s_idx)

    gpd = pgen.read(*cse, PGEN.GenosPhasingDosages)
    if cse[0] == "🥸":
        assert gpd is None
    else:
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
    last_end = 81265
    var_idxs = np.array([2], dtype=np.uint32)
    return cse, genos, phasing, dosages, last_end, var_idxs


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0, -1, 1], [1, -1, 0]], [[1, 0, -1], [1, 1, -1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0, 1], [1, 0, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan, 0.900024], [2.0, 1.0, np.nan]], np.float32)
    last_end = 81265
    var_idxs = np.arange(3, dtype=np.uint32)
    return cse, genos, phasing, dosages, last_end, var_idxs


@parametrize_with_cases(
    "cse, genos, phasing, dosages, last_end, var_idxs", cases=".", prefix="length_"
)
def test_chunk_with_length(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    last_end: int,
    var_idxs: np.uint32,
):
    mode = PGEN.GenosPhasingDosages
    max_mem = pgen._mem_per_variant(mode)
    gpd = pgen._chunk_ranges_with_length(*cse, max_mem, mode)
    for range_ in gpd:
        assert range_ is not None
        for chunk, end, v_idxs in range_:
            g, p, d = chunk
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_allclose(d, dosages, rtol=1e-5)
            assert end == last_end
            np.testing.assert_equal(v_idxs, var_idxs)


def n_vars_miss_chr():
    contig = "chr3"
    starts = 0
    ends = np.iinfo(np.int64).max
    desired = np.array([0], dtype=np.uint32)
    return contig, starts, ends, desired


def n_vars_all():
    contig = "chr1"
    starts = 0
    ends = np.iinfo(np.int64).max
    desired = np.array([3], dtype=np.uint32)
    return contig, starts, ends, desired


def n_vars_spanning_del():
    contig = "chr1"
    starts = 81262
    ends = 81263
    desired = np.array([1], dtype=np.uint32)
    return contig, starts, ends, desired


@parametrize_with_cases("contig, starts, ends, desired", cases=".", prefix="n_vars_")
def test_n_vars_in_ranges(
    pgen: PGEN,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    desired: NDArray[np.uint32],
):
    n_vars = pgen.n_vars_in_ranges(contig, starts, ends)
    assert n_vars == desired


def var_idxs_miss_chr():
    contig = "chr3"
    starts = 0
    ends = np.iinfo(np.int64).max
    desired = (np.array([], dtype=np.uint32), np.array([0, 0], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_all():
    contig = "chr1"
    starts = 0
    ends = np.iinfo(np.int64).max
    desired = (np.array([0, 1, 2], dtype=np.uint32), np.array([0, 3], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_spanning_del():
    contig = "chr1"
    starts = 81262
    ends = 81263
    desired = (np.array([0], dtype=np.uint32), np.array([0, 1], dtype=np.uint64))
    return contig, starts, ends, desired


@parametrize_with_cases("contig, starts, ends, desired", cases=".", prefix="var_idxs_")
def test_var_idxs(
    pgen: PGEN,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    desired: tuple[NDArray[np.uint32], NDArray[np.uint64]],
):
    var_idxs, offsets = pgen.var_idxs(contig, starts, ends)
    assert np.array_equal(var_idxs, desired[0])
    assert np.array_equal(offsets, desired[1])

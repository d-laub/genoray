from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pytest_cases import parametrize_with_cases

from genoray._pgen import PGEN, POS_TYPE, V_IDX_TYPE

tdir = Path(__file__).parent
ddir = tdir / "data"

N_SAMPLES = 2


def pgen_no_vzs():
    return PGEN(ddir / "biallelic.pgen")


def pgen_vzs():
    return PGEN(ddir / "biallelic.zst.pgen")


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
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    return cse, genos, phasing, dosages


def read_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    return cse, genos, phasing, dosages


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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
    np.testing.assert_equal(g, genos)

    d = pgen.read(*cse, PGEN.Dosages)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    g, p = pgen.read(*cse, PGEN.GenosPhasing)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)

    g, d = pgen.read(*cse, PGEN.GenosDosages)
    np.testing.assert_equal(g, genos)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)

    g, p, d = pgen.read(*cse, PGEN.GenosPhasingDosages)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p, phasing)
    np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    pgen: PGEN,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    n_variants = genos.shape[2]
    mode = PGEN.GenosPhasingDosages
    gpd = pgen.chunk(*cse, pgen._mem_per_variant(mode), mode)
    for i, (g, p, d) in enumerate(gpd):
        if n_variants != 0:
            np.testing.assert_equal(g, genos[..., [i]])
            np.testing.assert_equal(p, phasing[..., [i]])
            np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)
        else:
            np.testing.assert_equal(g, genos)
            np.testing.assert_equal(p, phasing)
            np.testing.assert_allclose(d, dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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

    (g, p, d), o = pgen.read_ranges(c, s, e, PGEN.GenosPhasingDosages)
    np.testing.assert_equal(g[..., o[0] : o[1]], genos)
    np.testing.assert_equal(g[..., o[1] : o[2]], genos)
    np.testing.assert_equal(p[..., o[0] : o[1]], phasing)
    np.testing.assert_equal(p[..., o[1] : o[2]], phasing)
    np.testing.assert_allclose(d[..., o[0] : o[1]], dosages, rtol=1e-5)
    np.testing.assert_allclose(d[..., o[1] : o[2]], dosages, rtol=1e-5)


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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

    n_variants = genos.shape[2]
    mode = PGEN.GenosPhasingDosages
    gpdo = pgen.chunk_ranges(c, s, e, max_mem=pgen._mem_per_variant(mode), mode=mode)
    for range_ in gpdo:
        for i, (g, p, d) in enumerate(range_):
            if n_variants != 0:
                np.testing.assert_equal(g, genos[..., [i]])
                np.testing.assert_equal(p, phasing[..., [i]])
                np.testing.assert_allclose(d, dosages[..., [i]], rtol=1e-5)
            else:
                np.testing.assert_equal(g, genos)
                np.testing.assert_equal(p, phasing)
                np.testing.assert_allclose(d, dosages, rtol=1e-5)


def samples_none():
    samples = None
    return samples


def samples_second():
    samples = "sample1"
    return samples


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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

    g, p, d = pgen.read(*cse, PGEN.GenosPhasingDosages)
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
    var_idxs = np.array([2], dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0, -1, 1], [1, -1, 0]], [[1, 0, -1], [1, 1, -1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0, 1], [1, 0, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan, 0.900024], [2.0, 1.0, np.nan]], np.float32)
    last_end = 81265
    var_idxs = np.arange(3, dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


def length_none():
    cse = "chr1", 0, 1
    # (s p v)
    genos, phasing, dosages = PGEN.GenosPhasingDosages.empty(N_SAMPLES, PGEN.ploidy, 0)
    # (s v)
    last_end = 1
    var_idxs = np.array([], dtype=V_IDX_TYPE)
    return cse, genos, phasing, dosages, last_end, var_idxs


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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


def n_vars_none():
    contig = "chr1"
    starts = 0
    ends = 1
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


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
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
    ends = np.iinfo(POS_TYPE).max
    desired = (np.array([], dtype=V_IDX_TYPE), np.array([0, 0], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_none():
    contig = "chr1"
    starts = 0
    ends = 1
    desired = (np.array([], dtype=V_IDX_TYPE), np.array([0, 0], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_all():
    contig = "chr1"
    starts = 0
    ends = np.iinfo(POS_TYPE).max
    desired = (np.array([0, 1, 2], dtype=V_IDX_TYPE), np.array([0, 3], dtype=np.uint64))
    return contig, starts, ends, desired


def var_idxs_spanning_del():
    contig = "chr1"
    starts = 81262
    ends = 81263
    desired = (np.array([0], dtype=V_IDX_TYPE), np.array([0, 1], dtype=np.uint64))
    return contig, starts, ends, desired


@parametrize_with_cases("pgen", cases=".", prefix="pgen_", scope="session")
@parametrize_with_cases("contig, starts, ends, desired", cases=".", prefix="var_idxs_")
def test_var_idxs(
    pgen: PGEN,
    contig: str,
    starts: ArrayLike,
    ends: ArrayLike,
    desired: tuple[NDArray[V_IDX_TYPE], NDArray[np.uint64]],
):
    var_idxs, offsets = pgen.var_idxs(contig, starts, ends)
    assert np.array_equal(var_idxs, desired[0])
    assert np.array_equal(offsets, desired[1])

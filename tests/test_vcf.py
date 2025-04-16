from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pytest_cases import fixture, parametrize_with_cases

from genoray import VCF
from genoray._utils import is_dtype

tdir = Path(__file__).parent
ddir = tdir / "data"


@fixture  # type: ignore
def vcf():
    return VCF(ddir / "biallelic.vcf.gz", dosage_field="DS")


def read_all():
    cse = "chr1", 81261, 81263
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
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    # (s p v)
    g = vcf.read(*cse, VCF.Genos8)
    assert g is not None
    assert is_dtype(g, np.int8)
    np.testing.assert_equal(g, genos)

    # (s p v)
    g = vcf.read(*cse, VCF.Genos16)
    assert g is not None
    assert is_dtype(g, np.int16)
    np.testing.assert_equal(g, genos)

    # (s p v)
    d = vcf.read(*cse, VCF.Dosages)
    assert d is not None
    np.testing.assert_equal(d, dosages)

    # (s p v)
    gd = vcf.read(*cse, VCF.Genos16Dosages)
    assert gd is not None
    g, d = gd
    assert is_dtype(g, np.int16)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(d, dosages)

    vcf.phasing = True
    gp = vcf.read(*cse)
    assert gp is not None
    assert is_dtype(gp, np.int16)
    g, p = np.array_split(gp, 2, 1)
    np.testing.assert_equal(g, genos)
    # (s 1 v) -> (s p v)
    np.testing.assert_equal(p.squeeze(1).astype(bool), phasing)

    gpd = vcf.read(*cse, VCF.Genos16Dosages)
    assert gpd is not None
    gp, d = gpd
    g, p = np.array_split(gp, 2, 1)
    np.testing.assert_equal(g, genos)
    np.testing.assert_equal(p.squeeze(1).astype(bool), phasing)
    np.testing.assert_equal(d, dosages)


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_chunk(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    vcf.phasing = True

    max_mem = vcf._mem_per_variant(VCF.Genos16Dosages)
    gpd = vcf.chunk(*cse, max_mem, VCF.Genos16Dosages)
    for i, chunk in enumerate(gpd):
        gp, d = chunk
        g, p = np.array_split(gp, 2, 1)
        p = p.squeeze(1).astype(bool)
        assert is_dtype(g, np.int16)
        assert is_dtype(d, np.float32)
        np.testing.assert_equal(g, genos[..., [i]])
        np.testing.assert_equal(p, phasing[..., [i]])
        np.testing.assert_equal(d, dosages[..., [i]])


def samples_none():
    samples = None
    return samples


def samples_second():
    samples = "sample1"
    return samples


@parametrize_with_cases("samples", cases=".", prefix="samples_")
@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    samples: ArrayLike | None,
):
    vcf.set_samples(samples)

    if samples is None:
        samples = vcf.available_samples
        s_idx = slice(None)
        s_sorter = slice(None)
    else:
        samples = np.atleast_1d(samples)
        s_idx = np.intersect1d(vcf.available_samples, samples, return_indices=True)[1]
        s_sorter = np.argsort(s_idx)

    assert vcf.current_samples == samples
    assert vcf.n_samples == len(samples)
    np.testing.assert_equal(vcf._s_sorter, s_sorter)

    vcf.phasing = True
    gpd = vcf.read(*cse, VCF.Genos16Dosages)
    assert gpd is not None
    gp, d = gpd
    g, p = np.array_split(gp, 2, 1)
    p = p.squeeze(1).astype(bool)
    np.testing.assert_equal(g, genos[s_idx])
    np.testing.assert_equal(p, phasing[s_idx])
    np.testing.assert_equal(d, dosages[s_idx])


def length_no_ext():
    cse = "chr1", 81264, 81265  # just 81265 in VCF
    # (s p v)
    genos = np.array([[[1], [0]], [[-1], [-1]]], np.int8)
    # (s v)
    phasing = np.array([[1], [0]], np.bool_)
    dosages = np.array([[0.9], [np.nan]], np.float32)
    last_end = 81265
    return cse, genos, phasing, dosages, last_end


def length_ext():
    cse = "chr1", 81262, 81263  # just 81263 in VCF
    # (s p v)
    genos = np.array([[[0, 1], [1, 0]], [[1, -1], [1, -1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 1], [1, 0]], np.bool_)
    dosages = np.array([[1.0, 0.9], [2.0, np.nan]], np.float32)
    last_end = 81265
    return cse, genos, phasing, dosages, last_end


@parametrize_with_cases("cse, genos, phasing, dosages, last_end", cases=".", prefix="length_")
def test_chunk_with_length(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
    last_end: int,
):
    vcf.phasing = True

    max_mem = vcf._mem_per_variant(VCF.Genos16Dosages)
    gpd = vcf.chunk_with_length(*cse, max_mem, VCF.Genos16Dosages)
    for i, (chunk, end) in enumerate(gpd):
        gp, d = chunk
        g, p = np.array_split(gp, 2, 1)
        p = p.squeeze(1).astype(bool)
        np.testing.assert_equal(g, genos)
        np.testing.assert_equal(p, phasing)
        np.testing.assert_equal(d, dosages)
        assert end == last_end

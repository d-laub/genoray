from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pytest_cases import fixture, parametrize_with_cases

from genoray import VCF
from genoray._utils import is_dtype

tdir = Path(__file__).parent
ddir = tdir / "data"


@fixture  # type: ignore
def vcf():
    return VCF(ddir / "test.vcf.gz", dosage_field="DS")


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


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="read_")
def test_set_samples(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    s_idx = [1]
    s_sorter = np.array([0], np.intp)
    samples = [vcf.available_samples[i] for i in s_idx]
    vcf.set_samples(samples)
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

def length_all():
    cse = "chr1", 81261, 81263
    # (s p v)
    genos = np.array([[[0, -1], [1, -1]], [[1, 0], [1, 1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0], [1, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan], [2.0, 1.0]], np.float32)
    return cse, genos, phasing, dosages


def length_spanning_del():
    cse = "chr1", 81262, 81263
    # (s p v)
    genos = np.array([[[0, -1], [1, -1]], [[1, 0], [1, 1]]], np.int8)
    # (s v)
    phasing = np.array([[1, 0], [1, 0]], np.bool_)
    dosages = np.array([[1.0, np.nan], [2.0, 1.0]], np.float32)
    return cse, genos, phasing, dosages


@parametrize_with_cases("cse, genos, phasing, dosages", cases=".", prefix="length_")
def test_chunk_with_length(
    vcf: VCF,
    cse: tuple[str, int, int],
    genos: NDArray[np.int8],
    phasing: NDArray[np.bool_],
    dosages: NDArray[np.float32],
):
    vcf.phasing = True

    max_mem = vcf._mem_per_variant(VCF.Genos16Dosages)
    gpd = vcf.chunk_with_length(*cse, max_mem, VCF.Genos16Dosages)
    for i, chunk in enumerate(gpd):
        gp, d = chunk
        g, p = np.array_split(gp, 2, 1)
        p = p.squeeze(1).astype(bool)
        np.testing.assert_equal(g, genos[..., [i]])
        np.testing.assert_equal(p, phasing[..., [i]])
        np.testing.assert_equal(d, dosages[..., [i]])
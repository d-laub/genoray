from __future__ import annotations

import numpy as np

from tests import _oracle
from tests.data.fixtures import FIXTURES


def _truth(name):
    return FIXTURES[name]().truth()


def test_genos_shape_and_values():
    truth = _truth("biallelic")
    g = _oracle.genos(truth, slice(None))
    assert g.shape == (2, 2, 6)
    np.testing.assert_array_equal(g[:, :, 0], np.array([[0, 1], [1, 1]]))
    np.testing.assert_array_equal(g[0, :, 1], np.array([-1, -1]))


def test_phasing_shape_and_values():
    truth = _truth("biallelic")
    p = _oracle.phasing(truth, slice(None))
    assert p.shape == (2, 6)
    assert p[0, 0] and p[1, 0]
    assert not p[1, 1]


def test_dosages_missing_is_nan():
    truth = _truth("biallelic")
    d = _oracle.dosages(truth, slice(None))
    assert d.shape == (2, 6)
    np.testing.assert_array_equal(d[:, 0], np.array([1.0, 2.0], np.float32))
    assert np.isnan(d[0, 1])


def test_index_subset():
    truth = _truth("biallelic")
    g = _oracle.genos(truth, [0, 2])
    assert g.shape == (2, 2, 2)


def test_split_phased_read_roundtrip():
    truth = _truth("biallelic")
    g = _oracle.genos(truth, slice(None)).astype(np.int8)
    p = _oracle.phasing(truth, slice(None))
    stacked = np.concatenate([g, p[:, None, :].astype(np.int8)], axis=1)
    g2, p2 = _oracle.split_phased(stacked)
    np.testing.assert_array_equal(g2, g)
    np.testing.assert_array_equal(p2, p)

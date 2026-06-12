import numpy as np
import pytest

from genoray._signatures import _cosine, _nnls


def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_is_zero():
    a = np.zeros(3)
    b = np.array([1.0, 2.0, 3.0])
    assert _cosine(a, b) == 0.0


def test_nnls_recovers_nonnegative_solution():
    # W h = m with a known nonnegative h
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    h_true = np.array([3.0, 5.0])
    m = W @ h_true
    h = _nnls(W, m)
    assert np.allclose(h, h_true, atol=1e-6)
    assert (h >= 0).all()

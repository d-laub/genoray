import numpy as np
import pytest

from genoray._signatures import _cosine, _fit_one, _nnls


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


def _toy_reference():
    # 4 mutation types, 3 signatures (columns sum to 1).
    W = np.array(
        [
            [0.7, 0.1, 0.25],
            [0.1, 0.7, 0.25],
            [0.1, 0.1, 0.25],
            [0.1, 0.1, 0.25],
        ]
    )
    return W


def test_fit_one_recovers_sparse_truth():
    W = _toy_reference()
    # Only signatures 0 and 1 are active; signature 2 absent.
    h_true = np.array([30.0, 70.0, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.01, min_activity=0.005)
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert h[2] == 0.0  # unused signature stays out
    assert h[0] == pytest.approx(30.0, rel=1e-3)
    assert h[1] == pytest.approx(70.0, rel=1e-3)


def test_fit_one_zero_sample():
    W = _toy_reference()
    h, cos = _fit_one(W, np.zeros(4), max_delta=0.01, min_activity=0.005)
    assert (h == 0).all()
    assert cos == 0.0


def test_fit_one_prunes_below_min_activity():
    W = _toy_reference()
    # Mostly signature 0, a tiny sliver of signature 1 (< 0.5% of total).
    h_true = np.array([100.0, 0.2, 0.0])
    m = W @ h_true
    h, cos = _fit_one(W, m, max_delta=0.001, min_activity=0.005)
    # signature 1 sliver is below min_activity -> pruned to 0
    assert h[1] == 0.0

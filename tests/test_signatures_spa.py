"""Unit and property tests for the SPA-faithful refit path."""

from __future__ import annotations

import numpy as np
import pytest

from genoray._signatures._common import _distance, _rel_l2, _round_conserve_sum


def test_rel_l2_perfect_reconstruction_is_zero():
    m = np.array([1.0, 2.0, 3.0])
    assert _rel_l2(m, m) == pytest.approx(0.0)


def test_rel_l2_zero_reconstruction_is_one():
    m = np.array([3.0, 4.0])
    assert _rel_l2(m, np.zeros(2)) == pytest.approx(1.0)


def test_rel_l2_zero_sample_is_zero_not_nan():
    m = np.zeros(4)
    assert _rel_l2(m, np.zeros(4)) == 0.0


def test_rel_l2_is_scale_invariant():
    m = np.array([1.0, 5.0, 2.0])
    recon = np.array([1.2, 4.5, 2.4])
    assert _rel_l2(m, recon) == pytest.approx(_rel_l2(10 * m, 10 * recon))


def test_distance_l2_matches_rel_l2():
    m = np.array([1.0, 5.0, 2.0])
    recon = np.array([1.2, 4.5, 2.4])
    assert _distance(m, recon, "l2") == pytest.approx(_rel_l2(m, recon))


def test_distance_cosine_is_lower_is_better():
    m = np.array([1.0, 2.0, 3.0])
    near = np.array([1.1, 2.0, 2.9])
    far = np.array([3.0, 2.0, 1.0])
    assert _distance(m, near, "cosine") < _distance(m, far, "cosine")
    assert _distance(m, m, "cosine") == pytest.approx(0.0)


def test_round_conserve_sum_conserves_total():
    x = np.array([10.4, 20.3, 69.3])
    out = _round_conserve_sum(x)
    assert out.sum() == pytest.approx(round(x.sum()))


def test_round_conserve_sum_returns_integers():
    x = np.array([10.4, 20.3, 69.3])
    out = _round_conserve_sum(x)
    assert np.all(out == np.floor(out))
    assert out.dtype == np.float64


def test_round_conserve_sum_keeps_zeros_zero():
    x = np.array([0.0, 50.5, 49.5])
    out = _round_conserve_sum(x)
    assert out[0] == 0.0


def test_round_conserve_sum_all_zero():
    out = _round_conserve_sum(np.zeros(5))
    assert out.sum() == 0.0
    assert np.all(out == 0.0)


@pytest.mark.parametrize("seed", range(30))
def test_round_conserve_sum_property(seed: int):
    """Conserves the rounded total and never goes negative, over random input."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    burden = float(rng.integers(1, 200_000))
    w = rng.random(n)
    x = w / w.sum() * burden
    out = _round_conserve_sum(x)
    assert out.sum() == pytest.approx(round(x.sum()))
    assert np.all(out >= 0.0), "roundConserveSum produced a negative activity"
    assert np.all(out == np.floor(out))

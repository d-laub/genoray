"""The forward-selection stop rule: ``criterion="cosine"`` vs ``criterion="bic"``.

Cosine similarity is scale-invariant, so thresholding its improvement makes the
``"cosine"`` rule blind to mutation burden -- a sample with 200,000 mutations gets
no more power to resolve a real signature than one with 200. The ``"bic"`` rule
compares a Poisson log-likelihood gain against a ``log(burden)`` penalty, so it
gains power as evidence accumulates.

These tests pin that structural difference on synthetic references. They are
deliberately not COSMIC-specific: the point is the stop rule, not any particular
signature set.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genoray._signatures import _fit_one, _poisson_ll, fit_signatures

#: Well-separated synthetic signatures, so "did the fit find the true set?" has an
#: unambiguous answer. Real COSMIC contains near-degenerate flat signatures
#: (SBS5/SBS40a/b/c) that no selection rule can fully disentangle; mixing that in
#: would test identifiability rather than the stop rule.
N_CHANNELS = 96
N_SIGS = 5


def _reference(seed: int = 7) -> np.ndarray:
    """``(N_CHANNELS, N_SIGS)`` column-normalized signatures, each with its own peak."""
    rng = np.random.default_rng(seed)
    block = N_CHANNELS // N_SIGS
    cols = []
    for k in range(N_SIGS):
        v = rng.random(N_CHANNELS) * 0.05 + 0.02
        v[k * block : (k + 1) * block] += 1.0
        cols.append(v)
    W = np.stack(cols, axis=1)
    return W / W.sum(axis=0)


def _catalogue(W: np.ndarray, burden: float, minor: float, rng) -> np.ndarray:
    """One Poisson sample: signature 0 dominant, signature 1 a weak ``minor`` share."""
    expected = W[:, [0, 1]] @ np.array([1.0 - minor, minor]) * burden
    return rng.poisson(expected).astype(np.float64)


def _hit_rate(criterion: str, burden: float, minor: float, reps: int = 20) -> int:
    """How many of ``reps`` samples recover BOTH true signatures."""
    W = _reference()
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(reps):
        m = _catalogue(W, burden, minor, rng)
        h, _cos = _fit_one(
            W, m, max_delta=0.01, min_activity=0.005, criterion=criterion
        )
        if {0, 1} <= set(np.nonzero(h)[0].tolist()):
            hits += 1
    return hits


def test_poisson_ll_is_maximized_at_the_truth():
    m = np.array([10.0, 20.0, 30.0])
    assert _poisson_ll(m, m) > _poisson_ll(m, np.array([20.0, 20.0, 20.0]))
    assert _poisson_ll(m, m) > _poisson_ll(m, np.array([1.0, 1.0, 1.0]))


def test_poisson_ll_tolerates_zero_expectation():
    """A signature can predict exactly zero in a channel; that must not be -inf."""
    value = _poisson_ll(np.array([0.0, 5.0]), np.array([0.0, 5.0]))
    assert np.isfinite(value)


def test_default_criterion_is_cosine():
    """The default must not change existing callers' results."""
    W = _reference()
    m = _catalogue(W, 5000.0, 0.05, np.random.default_rng(3))

    default_h, default_cos = _fit_one(W, m, max_delta=0.01, min_activity=0.005)
    explicit_h, explicit_cos = _fit_one(
        W, m, max_delta=0.01, min_activity=0.005, criterion="cosine"
    )

    np.testing.assert_array_equal(default_h, explicit_h)
    assert default_cos == explicit_cos


def test_cosine_rule_never_finds_a_weak_component_however_large_the_burden():
    """The defect: a 2%-activity signature stays invisible at 1000x the evidence.

    Cosine improvement is scale-invariant, so the 0.01 threshold rejects this
    component identically at 200 and at 200,000 mutations. More data buys nothing.
    """
    assert _hit_rate("cosine", burden=200, minor=0.02) == 0
    assert _hit_rate("cosine", burden=200_000, minor=0.02) == 0


def test_bic_rule_gains_power_as_burden_grows():
    """The fix: the same component becomes detectable once there is evidence for it."""
    low = _hit_rate("bic", burden=200, minor=0.02)
    mid = _hit_rate("bic", burden=2_000, minor=0.02)
    high = _hit_rate("bic", burden=20_000, minor=0.02)

    assert low < mid < high, f"expected monotone gain, got {low} -> {mid} -> {high}"
    assert high >= 18, f"bic should be near-certain at 20k mutations, got {high}/20"


def test_bic_beats_cosine_on_the_same_samples():
    for burden in (2_000, 20_000, 200_000):
        cosine = _hit_rate("cosine", burden=burden, minor=0.02)
        bic = _hit_rate("bic", burden=burden, minor=0.02)
        assert bic > cosine, f"burden {burden}: bic={bic} cosine={cosine}"


def test_bic_does_not_over_select_on_a_single_signature_sample():
    """A high-burden pure sample must still resolve to one signature."""
    W = _reference()
    rng = np.random.default_rng(5)
    m = rng.poisson(W[:, 0] * 50_000).astype(np.float64)

    h, cos = _fit_one(W, m, max_delta=0.01, min_activity=0.005, criterion="bic")

    assert np.nonzero(h)[0].tolist() == [0]
    assert cos > 0.99


def test_bic_on_an_empty_sample_returns_zero_activities():
    W = _reference()
    h, cos = _fit_one(
        W, np.zeros(N_CHANNELS), max_delta=0.01, min_activity=0.005, criterion="bic"
    )
    assert not h.any()
    assert cos == 0.0


def _frames(burden: float, minor: float, seed: int = 3):
    W = _reference()
    m = _catalogue(W, burden, minor, np.random.default_rng(seed))
    types = [f"T{i}" for i in range(N_CHANNELS)]
    catalogue = pl.DataFrame({"MutationType": types, "S1": m})
    reference = pl.DataFrame(
        {"MutationType": types, **{f"SIG{k}": W[:, k] for k in range(N_SIGS)}}
    )
    return catalogue, reference


def _n_selected(frame: pl.DataFrame) -> int:
    sigs = [c for c in frame.columns if c not in ("Sample", "cosine_similarity")]
    return int(sum(frame[c][0] > 0 for c in sigs))


def test_fit_signatures_threads_criterion_through_to_the_fit():
    catalogue, reference = _frames(burden=50_000, minor=0.02)

    cosine = fit_signatures(catalogue, reference, criterion="cosine")
    bic = fit_signatures(catalogue, reference, criterion="bic")

    assert _n_selected(bic) > _n_selected(cosine)
    assert bic["Sample"].to_list() == ["S1"]
    assert bic.columns == cosine.columns


def test_fit_signatures_defaults_to_cosine():
    catalogue, reference = _frames(burden=50_000, minor=0.02)
    assert fit_signatures(catalogue, reference).equals(
        fit_signatures(catalogue, reference, criterion="cosine")
    )


def test_fit_signatures_bic_is_invariant_to_n_jobs():
    catalogue, reference = _frames(burden=20_000, minor=0.05)
    serial = fit_signatures(catalogue, reference, criterion="bic", n_jobs=1)
    parallel = fit_signatures(catalogue, reference, criterion="bic", n_jobs=2)
    assert serial.equals(parallel)


def test_fit_signatures_rejects_an_unknown_criterion():
    catalogue, reference = _frames(burden=1_000, minor=0.05)
    with pytest.raises(ValueError, match="criterion must be"):
        fit_signatures(catalogue, reference, criterion="aic")  # type: ignore[arg-type]

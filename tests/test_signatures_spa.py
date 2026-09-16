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


def test_strategy_defaults_match_spa():
    from genoray import Spa

    s = Spa()
    assert s.metric == "l2"
    assert s.initial_remove_penalty == 0.05
    assert s.add_penalty == 0.05
    assert s.remove_penalty == 0.01
    assert s.background_sigs == ("SBS1", "SBS5")
    assert s.connected_sigs is True
    assert s.activity_scale == "burden"


def test_forward_defaults_match_current_behaviour():
    from genoray import Forward

    f = Forward()
    assert f.max_delta == 0.01
    assert f.min_activity == 0.005
    assert f.criterion == "cosine"


def test_strategies_are_frozen():
    import dataclasses

    from genoray import Forward, Spa

    with pytest.raises(dataclasses.FrozenInstanceError):
        Forward().max_delta = 0.2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Spa().metric = "cosine"  # type: ignore[misc]


def test_spa_rejects_unknown_metric():
    from genoray import Spa

    with pytest.raises(ValueError, match="metric"):
        Spa(metric="manhattan")  # type: ignore[arg-type]


def test_spa_rejects_unknown_activity_scale():
    from genoray import Spa

    with pytest.raises(ValueError, match="activity_scale"):
        Spa(activity_scale="fraction")  # type: ignore[arg-type]


def test_forward_rejects_unknown_criterion():
    from genoray import Forward

    with pytest.raises(ValueError, match="criterion"):
        Forward(criterion="aic")  # type: ignore[arg-type]


def test_spa_connected_groups_are_spas():
    from genoray._signatures._strategy import SPA_CONNECTED_GROUPS

    assert ("SBS2", "SBS13") in SPA_CONNECTED_GROUPS
    assert ("SBS7a", "SBS7b", "SBS7c", "SBS7d") in SPA_CONNECTED_GROUPS
    assert ("SBS10a", "SBS10b") in SPA_CONNECTED_GROUPS
    assert ("SBS17a", "SBS17b") in SPA_CONNECTED_GROUPS
    assert len(SPA_CONNECTED_GROUPS) == 4


def test_strategies_exported_from_genoray():
    import genoray

    assert "Forward" in genoray.__all__
    assert "Spa" in genoray.__all__
    assert genoray.Forward().criterion == "cosine"


def _toy_problem():
    """A 4-type, 3-signature toy catalogue with a known 2-signature answer."""
    import polars as pl

    ref = pl.DataFrame(
        {
            "MutationType": ["A", "B", "C", "D"],
            "S1": [1.0, 0.0, 0.0, 0.0],
            "S2": [0.0, 1.0, 0.0, 0.0],
            "S3": [0.0, 0.0, 1.0, 1.0],
        }
    )
    cat = pl.DataFrame(
        {"MutationType": ["A", "B", "C", "D"], "s1": [600.0, 400.0, 0.0, 0.0]}
    )
    return cat, ref


def test_strategy_forward_matches_legacy_kwargs():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    legacy = fit_signatures(cat, ref, max_delta=0.02, min_activity=0.01)
    viastrategy = fit_signatures(
        cat, ref, strategy=Forward(max_delta=0.02, min_activity=0.01)
    )
    assert legacy.equals(viastrategy)


def test_default_is_forward_and_unchanged():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    assert fit_signatures(cat, ref).equals(fit_signatures(cat, ref, strategy=Forward()))


def test_strategy_conflicts_with_legacy_kwarg():
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError, match="max_delta"):
        fit_signatures(cat, ref, strategy=Spa(), max_delta=0.02)


def test_strategy_conflict_names_every_offender():
    from genoray import Forward, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError) as exc:
        fit_signatures(cat, ref, strategy=Forward(), max_delta=0.02, min_activity=0.01)
    msg = str(exc.value)
    assert "max_delta" in msg and "min_activity" in msg


def test_passing_the_default_value_explicitly_still_conflicts():
    """The sentinel must distinguish 'not passed' from 'passed the default'."""
    from genoray import Spa, fit_signatures

    cat, ref = _toy_problem()
    with pytest.raises(ValueError, match="max_delta"):
        fit_signatures(cat, ref, strategy=Spa(), max_delta=0.01)


def _identity_ref(n_types: int = 5):
    """W = identity, so each signature owns exactly one mutation type."""
    return np.eye(n_types)


def test_exposure_burden_scale_sums_to_burden():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    m = np.array([300.0, 700.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    assert h.sum() == pytest.approx(1000.0)
    assert np.all(h == np.floor(h))


def test_exposure_raw_scale_is_unrounded():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    m = np.array([300.5, 699.5, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="raw")
    assert h[0] == pytest.approx(300.5)


def test_exposure_is_full_length_with_zeros_off_support():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(4)
    m = np.array([10.0, 20.0, 0.0, 0.0])
    h = _exposure(W, m, [0, 1], scale="burden")
    assert h.shape == (4,)
    assert h[2] == 0.0 and h[3] == 0.0


def test_exposure_zero_sample_returns_zeros():
    from genoray._signatures._spa import _exposure

    W = _identity_ref(3)
    h = _exposure(W, np.zeros(3), [0, 1, 2], scale="burden")
    assert np.all(h == 0.0)


def test_support_reads_nonzeros():
    from genoray._signatures._spa import _support

    assert _support(np.array([0.0, 5.0, 0.0, 2.0])) == [1, 3]


def test_remove_drops_a_signature_the_data_does_not_need():
    """S3 explains so little that dropping it costs less than the cutoff."""
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 10.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(W, m, h, cutoff=0.05, metric="l2", protected=frozenset())
    assert _support(out) == [0, 1]


def test_remove_keeps_a_signature_the_data_needs():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([400.0, 300.0, 300.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(W, m, h, cutoff=0.05, metric="l2", protected=frozenset())
    assert _support(out) == [0, 1, 2]


def test_remove_respects_protected():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 10.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset({2})
    )
    assert 2 in _support(out)


def test_protected_is_what_saves_the_signature():
    """The same sweep, same data: only the protected flag differs."""
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([600.0, 400.0, 10.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    unprotected = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset()
    )
    protected = _remove_all_single(
        W, m, h, cutoff=0.05, metric="l2", protected=frozenset({2})
    )
    assert _support(unprotected) == [0, 1]
    assert _support(protected) == [0, 1, 2]


def test_remove_never_empties_the_support():
    from genoray._signatures._spa import _exposure, _remove_all_single, _support

    W = _identity_ref(3)
    m = np.array([1000.0, 0.0, 0.0])
    h = _exposure(W, m, [0, 1, 2], scale="burden")
    out = _remove_all_single(W, m, h, cutoff=1.0, metric="l2", protected=frozenset())
    assert len(_support(out)) >= 1


@pytest.mark.parametrize("seed", range(20))
def test_removal_never_lowers_relative_l2(seed: int):
    """The claim that lets us skip SPA's negative-difference guard.

    NNLS over a column subset cannot beat NNLS over the superset, so dropping
    a column can only raise the relative L2 error.
    """
    from genoray._signatures._common import _nnls, _rel_l2

    rng = np.random.default_rng(seed)
    n_types, n_sigs = 24, 8
    W = rng.random((n_types, n_sigs))
    W = W / W.sum(axis=0)
    m = rng.random(n_types) * 1000.0

    full = list(range(n_sigs))
    base = _rel_l2(m, W[:, full] @ _nnls(W[:, full], m))
    for drop in full:
        sub = [i for i in full if i != drop]
        d = _rel_l2(m, W[:, sub] @ _nnls(W[:, sub], m))
        assert d >= base - 1e-9, f"dropping {drop} lowered relative L2"

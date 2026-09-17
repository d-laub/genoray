"""Ground-truth recovery: does strategy=Spa() find more true signatures?

These reproduce the measurement in issue #214 as a regression guard. They
assert the *ordering* of the two strategies, not the exact means, which are
Monte Carlo estimates over a modest number of reps.

Marked `network` because they fetch the real COSMIC reference.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from genoray import Forward, Spa, cosmic_signatures, fit_signatures

pytestmark = pytest.mark.network

N_REPS = 8
N_TRUE = 4

DISTINCT = ["SBS1", "SBS2", "SBS7a", "SBS17b"]
WITH_FLAT = ["SBS1", "SBS5", "SBS40a", "SBS2"]


def _synthetic_catalogue(ref: pl.DataFrame, truth: list[str], burden: int, seed: int):
    """Poisson counts from a known mixture of `truth` signatures."""
    rng = np.random.default_rng(seed)
    W = ref.select(truth).to_numpy()
    W = W / W.sum(axis=0)
    h = rng.dirichlet(np.ones(len(truth))) * burden
    counts = rng.poisson(W @ h).astype(np.float64)
    return pl.DataFrame({"MutationType": ref["MutationType"], "s": counts})


def _recovery(ref, truth, burden, strategy):
    """(mean true signatures found, mean false positives) over N_REPS."""
    found, false_pos = [], []
    for rep in range(N_REPS):
        cat = _synthetic_catalogue(ref, truth, burden, seed=1000 + rep)
        out = fit_signatures(cat, ref, strategy=strategy)
        sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
        selected = {c for c in sig_cols if out[c].item() > 0.0}
        found.append(len(selected & set(truth)))
        false_pos.append(len(selected - set(truth)))
    return float(np.mean(found)), float(np.mean(false_pos))


@pytest.fixture(scope="module")
def ref():
    return cosmic_signatures("SBS96")


@pytest.mark.parametrize("truth", [DISTINCT, WITH_FLAT], ids=["distinct", "with_flat"])
def test_spa_recovers_at_least_as_many_as_forward(ref, truth):
    burden = 100_000
    fwd, _ = _recovery(ref, truth, burden, Forward())
    spa, _ = _recovery(ref, truth, burden, Spa())
    assert spa >= fwd, f"Spa recovered {spa}/{N_TRUE}, Forward {fwd}/{N_TRUE}"


def test_spa_beats_forward_at_high_burden_on_distinct_signatures(ref):
    fwd, _ = _recovery(ref, DISTINCT, 100_000, Forward())
    spa, _ = _recovery(ref, DISTINCT, 100_000, Spa())
    assert spa > fwd, f"issue #214 measured 3.85 vs 3.50; got Spa {spa}, Forward {fwd}"


def test_spa_trades_precision_for_recall_at_low_burden(ref):
    """Spa is more sensitive and less precise than Forward. Guard both halves.

    Backward elimination from the saturated set recovers more true signatures
    than forward selection, and pays for it in false positives. Measured over
    24 configurations: Spa recovered +0.25 to +1.5 more truth, its false
    positives never exceeded 1.375, and on DISTINCT its false-positive count
    exceeded Forward's in 11 of 12 configurations. The upper bound guards
    against a regression where the prune stops working and Spa starts
    retaining everything.
    """
    fwd, fwd_fp = _recovery(ref, DISTINCT, 100, Forward())
    spa, spa_fp = _recovery(ref, DISTINCT, 100, Spa())
    assert spa > fwd, f"Spa recovered {spa}, Forward {fwd}"
    assert spa_fp <= 2.0, f"Spa false positives {spa_fp} — prune may be broken"


def test_spa_activities_conserve_burden_on_real_signatures(ref):
    cat = _synthetic_catalogue(ref, DISTINCT, 50_000, seed=42)
    out = fit_signatures(cat, ref, strategy=Spa())
    sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
    total = sum(out[c].item() for c in sig_cols)
    assert total == pytest.approx(round(cat["s"].sum()))


@pytest.mark.parametrize(
    ("seed", "expected", "why"),
    [
        (
            1003,
            ("SBS1", "SBS17b", "SBS2", "SBS7a"),
            "stage 3 removes a spurious SBS5, leaving exactly the ground truth",
        ),
        (
            1023,
            ("SBS1", "SBS17b", "SBS2", "SBS5", "SBS7a"),
            "stage 3 adds SBS2, a true signature the backward prune had dropped",
        ),
        (
            1010,
            ("SBS17b", "SBS2", "SBS7a", "SBS7b"),
            "stage 3 adds SBS7b and lets the sweep drop a spurious SBS5",
        ),
    ],
)
def test_stage_three_refinement_changes_the_result(ref, seed, expected, why):
    """Pin cases where the add-remove refinement loop is load-bearing.

    Aggregate recovery statistics cannot see stage 3 — the saturate-then-prune
    stages do almost all the work, and stubbing the refinement loop moves mean
    recovery by at most 0.125 signatures. These three samples are cases where
    it changes the answer (83 of 320 fits differ under the stub), so they are
    the only guard the loop has.
    """
    cat = _synthetic_catalogue(ref, DISTINCT, 100, seed=seed)
    out = fit_signatures(cat, ref, strategy=Spa())
    sig_cols = [c for c in out.columns if c not in ("Sample", "cosine_similarity")]
    selected = tuple(sorted(c for c in sig_cols if out[c].item() > 0.0))
    assert selected == expected, why

"""Greedy forward selection from the empty signature set. genoray's own algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ._common import _cosine, _nnls, _poisson_ll

if TYPE_CHECKING:  # pragma: no cover
    from ._strategy import Forward

#: Forward-selection stop rules. See ``fit_signatures``.
Criterion = Literal["cosine", "bic"]


def _fit_one(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    *,
    max_delta: float,
    min_activity: float,
    criterion: Criterion = "cosine",
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by sparse forward selection.

    Returns ``(activities, cosine)`` where ``activities`` has one entry per
    reference signature (column of ``W``), zero for unselected signatures, and
    ``cosine`` is the reconstruction cosine similarity of the final fit. The
    returned cosine is the fit's reconstruction quality under either criterion;
    only the *stop rule* changes.
    """
    n_sigs = W.shape[1]
    full = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return full, 0.0

    active: list[int] = []
    remaining = list(range(n_sigs))
    best_cos = 0.0
    best_ll = -np.inf
    # BIC observation count: each mutation is one Poisson event, so the penalty
    # for one extra free parameter is log(total burden). Clamped at 2 so a
    # 1-mutation sample gets a non-zero penalty instead of log(1) == 0, which
    # would accept every candidate.
    log_n_obs = float(np.log(max(float(np.sum(m)), 2.0)))

    # Forward selection: add the signature that most improves the fit, until the
    # improvement no longer clears the stop criterion.
    while remaining:
        best = None  # (score, sig_index, cosine)
        for c in remaining:
            cand = active + [c]
            h_sub = _nnls(W[:, cand], m)
            recon = W[:, cand] @ h_sub
            cos = _cosine(m, recon)
            score = cos if criterion == "cosine" else _poisson_ll(m, recon)
            if best is None or score > best[0]:
                best = (score, c, cos)
        assert best is not None
        score, c, cos = best
        if criterion == "cosine":
            if score - best_cos < max_delta:
                break
            best_cos = score
        else:
            # One more signature is one more free parameter. Accept it only when
            # twice the log-likelihood gain clears the BIC penalty. Unlike the
            # cosine delta, this scales with burden: more mutations buy more
            # power to resolve a real but low-activity signature.
            if 2.0 * (score - best_ll) < log_n_obs:
                break
            best_ll = score
        active.append(c)
        remaining.remove(c)

    if not active:
        return full, 0.0

    # Prune signatures below min_activity (as a fraction of total), re-fitting
    # survivors until the active set is stable.
    while True:
        h_sub = _nnls(W[:, active], m)
        total = float(h_sub.sum())
        if total == 0.0:
            return full, 0.0
        keep = [
            active[i] for i in range(len(active)) if h_sub[i] / total >= min_activity
        ]
        if len(keep) == len(active):
            break
        if not keep:
            # everything pruned: keep the single largest contributor
            keep = [active[int(np.argmax(h_sub))]]
        active = keep

    h_sub = _nnls(W[:, active], m)
    recon = W[:, active] @ h_sub
    cos = _cosine(m, recon)
    for i, sig in enumerate(active):
        full[sig] = h_sub[i]
    return full, cos


def _fit_one_forward(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    spec: "Forward",
) -> tuple[NDArray[np.float64], float]:
    """Adapt ``_fit_one`` to the strategy-object calling convention."""
    return _fit_one(
        W,
        m,
        max_delta=spec.max_delta,
        min_activity=spec.min_activity,
        criterion=spec.criterion,
    )

"""SigProfilerAssignment's ``cosmic_fit``, reimplemented.

Backward elimination from the saturated signature set, then add-remove
refinement layers. Transcribed from SigProfilerAssignment ``main``:
``single_sample.py`` (``fit_signatures``, ``add_signatures``,
``remove_all_single_signatures``, ``add_remove_signatures``,
``add_connected_sigs``) and ``decompose_subroutines.py``
(``process_sample``), ``solver="nnls"`` and ``pcawg_rule=False`` only.

Like SPA, the state carried between stages is a full-length exposure
*vector*, not an index set, and the active set is read off its nonzeros.
This matters: both of SPA's inner routines round what they record, so a
signature whose rescaled activity falls below 0.5 drops out of the support
without any threshold testing it. Distances, however, are always computed
from the unrounded NNLS reconstruction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._common import _distance, _nnls, _round_conserve_sum


def _support(h: NDArray[np.floating]) -> list[int]:
    """The active signature indices: the nonzeros of an exposure vector."""
    return [int(i) for i in np.nonzero(h)[0]]


def _exposure(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    active: Sequence[int],
    *,
    scale: str,
) -> NDArray[np.float64]:
    """NNLS over ``active``, scattered into a full-length exposure vector.

    With ``scale="burden"`` the weights are renormalized to sum to the
    sample's mutation count and integer-rounded conserving that sum, which is
    how SPA reports every intermediate and final exposure. With
    ``scale="raw"`` the raw NNLS weights are returned.
    """
    full = np.zeros(W.shape[1], dtype=np.float64)
    active = sorted(active)
    if not active:
        return full
    weights = _nnls(W[:, active], m)
    total = float(weights.sum())
    if total == 0.0:
        return full
    if scale == "burden":
        burden = float(np.sum(m))
        weights = _round_conserve_sum(weights / total * burden)
    full[active] = weights
    return full


def _reconstruction(
    W: NDArray[np.floating], m: NDArray[np.floating], active: Sequence[int]
) -> NDArray[np.float64]:
    """The *unrounded* NNLS reconstruction over ``active``. Scoring uses this.

    SPA computes every distance from ``np.dot(W1, weights)`` with raw weights,
    even where the exposures it records alongside are rounded. Scoring off the
    rounded vector would be a divergence.
    """
    active = sorted(active)
    if not active:
        return np.zeros(W.shape[0], dtype=np.float64)
    return W[:, active] @ _nnls(W[:, active], m)


def _remove_all_single(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    h: NDArray[np.floating],
    *,
    cutoff: float,
    metric: str,
    protected: frozenset[int],
) -> NDArray[np.float64]:
    """SPA's ``remove_all_single_signatures``.

    Repeatedly removes the signature whose removal degrades the distance
    least, while that degradation stays at or below ``cutoff``. Stops when the
    best available removal costs more than ``cutoff``, or when one signature
    is left.

    The cutoff is measured against the *current* fit, not the one this sweep
    started from: SPA advances its baseline after every accepted layer.
    Measuring against the original baseline instead makes the prune far too
    permissive.
    """
    active = _support(h)
    if len(active) <= 1:
        return np.asarray(h, dtype=np.float64).copy()

    base = _distance(m, _reconstruction(W, m, active), metric)
    scale = "burden"

    while len(active) > 1:
        best_d = np.inf
        best_active: list[int] | None = None
        for i in active:
            if i in protected:
                continue
            cand = [j for j in active if j != i]
            d = _distance(m, _reconstruction(W, m, cand), metric)
            if d < best_d:
                best_d = d
                best_active = cand
        if best_active is None:
            break  # every remaining signature is protected
        if best_d - base > cutoff:
            break
        active = best_active
        base = best_d

    return _exposure(W, m, active, scale=scale)


def _try_add(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    active: Sequence[int],
    cand: int,
    *,
    cutoff: float,
    metric: str,
) -> list[int]:
    """SPA's ``add_signatures`` restricted to a single candidate.

    ``add_remove_signatures`` always calls ``add_signatures`` with
    ``toBeAdded=[c]``, which restricts its candidate pool to ``c`` alone. Its
    loop therefore runs at most one accepting iteration, and the whole
    routine collapses to: add ``c`` if and only if it improves the distance
    by strictly more than ``cutoff``.

    The strict inequality is SPA's (``if originalSimilarity - bestSimilarity >
    cutoff``). An empty active set has infinite distance, so the first
    signature is always accepted, matching SPA's ``originalSimilarity =
    np.inf`` initialization.
    """
    active = sorted(active)
    if cand in active:
        return active
    base = np.inf if not active else _distance(m, _reconstruction(W, m, active), metric)
    new = sorted([*active, cand])
    d_new = _distance(m, _reconstruction(W, m, new), metric)
    return new if base - d_new > cutoff else active


def _expand_connected(
    active: Sequence[int], groups: tuple[tuple[int, ...], ...]
) -> list[int]:
    """SPA's ``add_connected_sigs``: if any group member is active, add them all."""
    out = set(active)
    for group in groups:
        if out.intersection(group):
            out.update(group)
    return sorted(out)

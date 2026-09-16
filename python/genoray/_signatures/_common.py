"""Numeric primitives shared by the refit strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls


def _cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine similarity of two vectors; 0.0 if either has zero norm."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nnls(W: NDArray[np.floating], m: NDArray[np.floating]) -> NDArray[np.float64]:
    """Non-negative least squares: argmin_{h>=0} ||W h - m||."""
    h, _ = nnls(W.astype(np.float64), m.astype(np.float64))
    return h


def _poisson_ll(m: NDArray[np.floating], e: NDArray[np.floating]) -> float:
    """Poisson log-likelihood of observed counts ``m`` under expected counts ``e``.

    Drops the ``-log(m!)`` term. It depends only on the data, so it cancels in
    every likelihood *difference* -- which is all the ``"bic"`` criterion uses.
    """
    e = np.clip(np.asarray(e, dtype=np.float64), 1e-12, None)
    return float(np.sum(np.asarray(m, dtype=np.float64) * np.log(e) - e))


def _rel_l2(m: NDArray[np.floating], recon: NDArray[np.floating]) -> float:
    """Relative L2 error ``||m - recon||_2 / ||m||_2``; 0.0 for a zero sample.

    This is SigProfilerAssignment's ``metric="l2"`` score. It is
    scale-invariant, and because the denominator is constant across candidate
    fits of the same sample, it orders candidates identically to the raw
    residual norm.
    """
    nm = float(np.linalg.norm(m))
    if nm == 0.0:
        return 0.0
    return float(np.linalg.norm(np.asarray(m, dtype=np.float64) - recon) / nm)


def _distance(
    m: NDArray[np.floating], recon: NDArray[np.floating], metric: str
) -> float:
    """Lower-is-better distance between a sample and its reconstruction.

    Both branches are lower-is-better so every comparison in the SPA path is
    written once, exactly as SigProfilerAssignment writes it.
    """
    if metric == "l2":
        return _rel_l2(m, recon)
    return 1.0 - _cosine(m, recon)


def _round_conserve_sum(x: NDArray[np.floating]) -> NDArray[np.float64]:
    """Round to integers while conserving the total. SPA's ``roundConserveSum``.

    Ceil every entry, then decrement the entries that were rounded up the most
    until the total matches ``round(sum(x))``. Transcribed from
    SigProfilerAssignment so activities agree entry for entry.
    """
    x = np.asarray(x, dtype=np.float64)
    total = np.round(np.sum(x))
    x_out = np.ceil(x)
    # x - x_out is in (-1, 0]; ascending order puts the largest round-up first.
    order = np.argsort(x - x_out)
    n_to_drop = int(np.sum(x_out) - total + 1e-10)
    if n_to_drop > 0:
        x_out[order[:n_to_drop]] -= 1
    return x_out

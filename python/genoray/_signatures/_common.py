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

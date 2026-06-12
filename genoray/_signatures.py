"""COSMIC mutational-signature refitting.

Ports the core of SigProfilerAssignment: a sparse forward-selection refit that
decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from ._mutcat import labels  # noqa: F401  # used in later tasks

Kind = Literal["SBS96", "DBS78", "ID83"]


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


def _fit_one(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    *,
    max_delta: float,
    min_activity: float,
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by sparse forward selection.

    Returns ``(activities, cosine)`` where ``activities`` has one entry per
    reference signature (column of ``W``), zero for unselected signatures, and
    ``cosine`` is the reconstruction cosine similarity of the final fit.
    """
    n_sigs = W.shape[1]
    full = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return full, 0.0

    active: list[int] = []
    remaining = list(range(n_sigs))
    best_cos = 0.0

    # Forward selection: add the signature that most improves cosine, until the
    # improvement falls below max_delta.
    while remaining:
        best = None  # (cos, sig_index, h_subvector)
        for c in remaining:
            cand = active + [c]
            h_sub = _nnls(W[:, cand], m)
            recon = W[:, cand] @ h_sub
            cos = _cosine(m, recon)
            if best is None or cos > best[0]:
                best = (cos, c, h_sub)
        assert best is not None
        cos, c, _ = best
        if cos - best_cos < max_delta:
            break
        best_cos = cos
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

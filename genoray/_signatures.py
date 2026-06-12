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

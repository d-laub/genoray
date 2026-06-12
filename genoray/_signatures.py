"""COSMIC mutational-signature refitting.

Ports the core of SigProfilerAssignment: a sparse forward-selection refit that
decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
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


def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
) -> pl.DataFrame:
    """Refit a mutation catalogue against reference signatures.

    Parameters
    ----------
    catalogue
        A ``mutation_matrix``-shaped frame: a ``MutationType`` column followed by
        one numeric column per sample.
    reference
        A ``MutationType`` column followed by one column per reference signature.
        Columns need not be pre-normalized; each is scaled to sum 1 so reported
        activities are in mutation-count units.
    max_delta
        Minimum cosine-similarity improvement to keep adding a signature
        (forward-selection stop criterion).
    min_activity
        Minimum fractional contribution; signatures below this are pruned.

    Returns
    -------
    pl.DataFrame
        One row per sample: a ``Sample`` column, one Float column per reference
        signature (activities, 0.0 if unselected), and a ``cosine_similarity``
        column for the final reconstruction.

    Raises
    ------
    ValueError
        If a ``MutationType`` present in the catalogue is missing from the
        reference (rows cannot be aligned).
    """
    if "MutationType" not in catalogue.columns:
        raise ValueError("catalogue must have a 'MutationType' column.")
    if "MutationType" not in reference.columns:
        raise ValueError("reference must have a 'MutationType' column.")

    sample_cols = [c for c in catalogue.columns if c != "MutationType"]
    sig_cols = [c for c in reference.columns if c != "MutationType"]

    # Align reference rows to the catalogue's row order by joining on MutationType.
    aligned = catalogue.select("MutationType").join(
        reference, on="MutationType", how="left"
    )
    missing = aligned.filter(pl.col(sig_cols[0]).is_null())
    if missing.height > 0:
        bad = missing["MutationType"].to_list()
        raise ValueError(
            f"reference is missing MutationType rows present in the catalogue: {bad}"
        )

    W = aligned.select(sig_cols).to_numpy().astype(np.float64)  # (n_types, n_sigs)
    col_sums = W.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0  # avoid div-by-zero for empty signatures
    W = W / col_sums  # normalize each signature column to sum 1

    M = (
        catalogue.select(sample_cols).to_numpy().astype(np.float64)
    )  # (n_types, n_samples)

    activities = np.zeros((len(sample_cols), len(sig_cols)), dtype=np.float64)
    cosines = np.zeros(len(sample_cols), dtype=np.float64)
    for j in range(len(sample_cols)):
        h, cos = _fit_one(W, M[:, j], max_delta=max_delta, min_activity=min_activity)
        activities[j] = h
        cosines[j] = cos

    out: dict[str, object] = {"Sample": sample_cols}
    for i, sig in enumerate(sig_cols):
        out[sig] = activities[:, i]
    out["cosine_similarity"] = cosines
    return pl.DataFrame(out)

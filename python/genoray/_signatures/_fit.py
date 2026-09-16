"""The ``fit_signatures`` driver: alignment, strategy dispatch, parallel fan-out."""

from __future__ import annotations

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from ._forward import Criterion, _fit_one


def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    max_delta: float = 0.01,
    min_activity: float = 0.005,
    criterion: Criterion = "cosine",
    n_jobs: int = 1,
    backend: str = "loky",
) -> pl.DataFrame:
    """Refit a mutation catalogue against reference signatures.

    Args:
        catalogue: A ``mutation_matrix``-shaped frame: a ``MutationType`` column followed by
            one numeric column per sample.
        reference: A ``MutationType`` column followed by one column per reference signature.
            Columns need not be pre-normalized; each is scaled to sum 1 so reported
            activities are in mutation-count units.
        max_delta: Minimum cosine-similarity improvement to keep adding a signature.
            Only used when ``criterion="cosine"``; ignored otherwise.
        min_activity: Minimum fractional contribution; signatures below this are pruned.
        criterion: Forward-selection stop rule.

            ``"cosine"`` (default) stops when the best candidate improves cosine
            similarity by less than ``max_delta``. Cosine is scale-invariant, so this
            threshold is blind to mutation burden: a sample with 100,000 mutations
            gets no more power to resolve a real signature than one with 100. On
            synthetic mixtures of four known COSMIC signatures it plateaus around 3.3
            of 4 recovered and does **not** improve with burden -- and around 2.6 of 4
            when the mixture includes the flat-spectrum family (SBS5, SBS40a), whose
            contribution is what a too-coarse threshold absorbs first.

            ``"bic"`` stops when twice the Poisson log-likelihood gain fails to clear
            the Bayesian information criterion penalty ``log(total burden)`` for the
            one added free parameter. The likelihood gain grows with burden while the
            penalty grows only logarithmically, so the rule is burden-aware and
            consistent: on the same synthetic mixtures it recovers 4.00 of 4 from
            ~1,000 mutations upward (~10,000 with flat signatures present) with
            0.00-0.03 false positives per sample.

            The trade-off runs the other way at very low burden, where ``"bic"``
            over-selects (~1.4 false positives per sample at 100 mutations). The
            default stays ``"cosine"`` for backward compatibility; prefer ``"bic"``
            for whole-genome catalogues.
        n_jobs: Number of parallel workers for the per-sample refit (passed to
            ``joblib.Parallel``). ``1`` (default) runs serially; ``-1`` uses all
            cores. Results are identical regardless of ``n_jobs``.
        backend: ``joblib`` backend (default ``"loky"``, process-based). Samples are
            refit independently, so a process backend avoids GIL contention from
            the forward-selection orchestration.

    Returns:
        pl.DataFrame: One row per sample: a ``Sample`` column, one Float column per reference
        signature (activities, 0.0 if unselected), and a ``cosine_similarity``
        column for the final reconstruction.

    Raises:
        ValueError: If ``criterion`` is not one of ``"cosine"`` / ``"bic"``, or if a
            ``MutationType`` present in the catalogue is missing from the reference
            (rows cannot be aligned).
    """
    if criterion not in ("cosine", "bic"):
        raise ValueError(f"criterion must be 'cosine' or 'bic', got {criterion!r}.")
    if "MutationType" not in catalogue.columns:
        raise ValueError("catalogue must have a 'MutationType' column.")
    if "MutationType" not in reference.columns:
        raise ValueError("reference must have a 'MutationType' column.")

    sample_cols = [c for c in catalogue.columns if c != "MutationType"]
    sig_cols = [c for c in reference.columns if c != "MutationType"]

    # Align reference rows to the catalogue's row order by joining on MutationType.
    # maintain_order="left" is required: without it, Polars' hash-join is
    # non-deterministic under concurrent workloads and can reorder left-frame rows.
    aligned = catalogue.select("MutationType").join(
        reference, on="MutationType", how="left", maintain_order="left"
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
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_fit_one)(
            W,
            M[:, j],
            max_delta=max_delta,
            min_activity=min_activity,
            criterion=criterion,
        )
        for j in range(len(sample_cols))
    )
    for j, (h, cos) in enumerate(results):
        activities[j] = h
        cosines[j] = cos

    out: dict[str, object] = {"Sample": sample_cols}
    for i, sig in enumerate(sig_cols):
        out[sig] = activities[:, i]
    out["cosine_similarity"] = cosines
    return pl.DataFrame(out)

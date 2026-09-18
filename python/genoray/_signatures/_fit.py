"""The ``fit_signatures`` driver: alignment, strategy dispatch, parallel fan-out."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from genoray._unset import _UNSET
from ._forward import Criterion, _fit_one_forward
from ._spa import _fit_one_spa
from ._strategy import SPA_CONNECTED_GROUPS, Forward, Spa, Strategy

#: Real defaults for the legacy shorthand arguments. Kept here so the
#: docstring and the resolution agree in one place.
_FORWARD_DEFAULTS = Forward()


def _resolve_strategy(
    strategy: Strategy | None,
    max_delta: Any,
    min_activity: Any,
    criterion: Any,
) -> Strategy:
    """Reconcile ``strategy=`` with the legacy flat keyword arguments.

    The flat arguments are permanent shorthand for the forward path, not
    deprecated. Combining them with an explicit ``strategy`` is an error
    rather than a silent drop.
    """
    passed = {
        "max_delta": max_delta,
        "min_activity": min_activity,
        "criterion": criterion,
    }
    explicit = [name for name, val in passed.items() if val is not _UNSET]

    if strategy is None:
        return Forward(
            max_delta=(
                _FORWARD_DEFAULTS.max_delta if max_delta is _UNSET else max_delta
            ),
            min_activity=(
                _FORWARD_DEFAULTS.min_activity
                if min_activity is _UNSET
                else min_activity
            ),
            criterion=(
                _FORWARD_DEFAULTS.criterion if criterion is _UNSET else criterion
            ),
        )

    if explicit:
        raise ValueError(
            f"strategy= was given together with {', '.join(sorted(explicit))}. "
            "The flat arguments are shorthand for the forward path; pass them "
            "inside Forward(...) instead, or drop strategy=."
        )
    return strategy


def _resolve_sig_names(
    sig_cols: list[str], spec: Spa
) -> tuple[frozenset[int], tuple[tuple[int, ...], ...]]:
    """Map ``Spa``'s signature *names* onto column indices of the reference.

    Names absent from the reference are dropped silently, matching SPA's
    ``get_indeces``. That is what makes the default ``("SBS1", "SBS5")``
    background and the SBS-only connected groups inert for DBS78 and ID83
    without a special case.

    Resolution happens once, here, rather than per sample: it needs the
    reference's column names, and the fan-out below would otherwise repeat it
    once per sample.
    """
    index_of = {name: i for i, name in enumerate(sig_cols)}

    protected = frozenset(
        index_of[n] for n in (spec.background_sigs or ()) if n in index_of
    )

    if isinstance(spec.connected_sigs, (bool, np.bool_)):
        raw_groups: Sequence[Sequence[str]] = (
            SPA_CONNECTED_GROUPS if spec.connected_sigs else ()
        )
    else:
        raw_groups = spec.connected_sigs

    groups = tuple(
        tuple(sorted(index_of[n] for n in g if n in index_of)) for g in raw_groups
    )
    # A group with fewer than two present members can never expand anything.
    groups = tuple(g for g in groups if len(g) > 1)
    return protected, groups


def fit_signatures(
    catalogue: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    strategy: Strategy | None = None,
    max_delta: float = _UNSET,
    min_activity: float = _UNSET,
    criterion: Criterion = _UNSET,
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
        strategy: The refit algorithm and its parameters: :class:`Forward` (the
            default, ``Forward()``) or :class:`Spa`. Cannot be combined with
            ``max_delta``, ``min_activity``, or ``criterion``.

            :class:`Spa` reimplements SigProfilerAssignment's ``cosmic_fit``:
            saturate over every reference signature, prune backward on
            relative L2 error, then refine with add-remove layers (with
            SBS1/SBS5 background protection and connected-signature groups
            on by default). Under its default ``activity_scale="burden"``,
            the signature columns hold integer-valued floats that sum to
            each sample's mutation burden, unlike the forward path (and
            ``Spa(activity_scale="raw")``), which report raw NNLS weights
            that need not sum to anything in particular.
        max_delta: Shorthand for ``strategy=Forward(max_delta=...)``. Minimum
            cosine-similarity improvement to keep adding a signature. Only used
            when ``criterion="cosine"``; ignored otherwise. Default ``0.01``.
            Cannot be combined with ``strategy=``.
        min_activity: Shorthand for ``strategy=Forward(min_activity=...)``.
            Minimum fractional contribution; signatures below this are pruned.
            Default ``0.005``. Cannot be combined with ``strategy=``.
        criterion: Shorthand for ``strategy=Forward(criterion=...)``.
            Forward-selection stop rule. Default ``"cosine"``. Cannot be
            combined with ``strategy=``.

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
        ValueError: If ``criterion`` is not one of ``"cosine"`` / ``"bic"``, if
            ``strategy=`` is combined with any of ``max_delta``, ``min_activity``,
            or ``criterion``, or if a ``MutationType`` present in the catalogue is
            missing from the reference (rows cannot be aligned).
    """
    spec = _resolve_strategy(strategy, max_delta, min_activity, criterion)
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
    if isinstance(spec, Forward):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_one_forward)(W, M[:, j], spec) for j in range(len(sample_cols))
        )
    else:
        protected, groups = _resolve_sig_names(sig_cols, spec)
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_fit_one_spa)(W, M[:, j], spec, protected=protected, groups=groups)
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

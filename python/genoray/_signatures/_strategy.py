"""Refit strategies.

A strategy both selects the algorithm and carries its parameters. Each one
holds only the knobs its own algorithm uses, so a forward-selection threshold
cannot reach backward elimination: that is a type error rather than a
silently ignored keyword argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._forward import Criterion

#: Score used by the SPA path. ``"l2"`` is SigProfilerAssignment's default.
Metric = Literal["l2", "cosine"]

#: How the SPA path reports activities.
ActivityScale = Literal["burden", "raw"]

#: SigProfilerAssignment's ``add_connected_sigs`` groups. Whenever any member
#: is selected, every member is force-added. SBS-only; inert for DBS78/ID83.
SPA_CONNECTED_GROUPS: tuple[tuple[str, ...], ...] = (
    ("SBS2", "SBS13"),
    ("SBS7a", "SBS7b", "SBS7c", "SBS7d"),
    ("SBS10a", "SBS10b"),
    ("SBS17a", "SBS17b"),
)


@dataclass(frozen=True)
class Forward:
    """Greedy forward selection from the empty set. genoray's own algorithm.

    Args:
        max_delta: Minimum cosine-similarity improvement to keep adding a
            signature. Used only when ``criterion="cosine"``.
        min_activity: Minimum fractional contribution; signatures below this
            are pruned after selection.
        criterion: Stop rule. ``"cosine"`` is scale-invariant and therefore
            blind to mutation burden. ``"bic"`` is burden-aware and
            consistent, but over-selects below ~1,000 mutations.
    """

    max_delta: float = 0.01
    min_activity: float = 0.005
    criterion: Criterion = "cosine"

    def __post_init__(self) -> None:
        if self.criterion not in ("cosine", "bic"):
            raise ValueError(
                f"criterion must be 'cosine' or 'bic', got {self.criterion!r}."
            )


@dataclass(frozen=True)
class Spa:
    """Backward elimination from the saturated set: SigProfilerAssignment's ``cosmic_fit``.

    Every default is SigProfilerAssignment's own, so ``Spa()`` is
    ``cosmic_fit`` as shipped. Recovers more true signatures than ``Forward``
    at every burden measured, but is also scale-invariant and so is not a
    consistent estimator either.

    Args:
        metric: Score. ``"l2"`` is relative L2 error, SPA's default.
            ``"cosine"`` diverges from SPA -- see the class notes below.
        initial_remove_penalty: Cutoff for the prune that follows the
            saturated fit. Background signatures are **not** protected here.
        add_penalty: A candidate must improve the distance by strictly more
            than this to be added during refinement.
        remove_penalty: Cutoff for the removal sweeps during refinement.
        background_sigs: Signature names always kept once refinement starts.
            Names absent from the reference are ignored, which is what makes
            the default inert for DBS78 and ID83. ``None`` disables.
        connected_sigs: Force-add co-occurring partners. ``True`` uses
            :data:`SPA_CONNECTED_GROUPS`; a sequence of groups supplies your
            own; ``False`` disables.
        activity_scale: ``"burden"`` rescales and integer-rounds activities so
            they sum to the sample's mutation count, as SPA reports them.
            ``"raw"`` returns unscaled NNLS weights, as the forward path does.

    Notes:
        Two SPA behaviours are deliberately not reproduced. SPA refuses to
        remove a signature whose removal *improves* the fit; on the default
        ``metric="l2"`` path that branch is unreachable, because NNLS over a
        column subset cannot beat NNLS over the superset, so omitting it
        cannot change a result. Under ``metric="cosine"`` it is reachable and
        this class diverges. SPA also draws from the global RNG for values it
        never uses; this implementation is deterministic.
    """

    metric: Metric = "l2"
    initial_remove_penalty: float = 0.05
    add_penalty: float = 0.05
    remove_penalty: float = 0.01
    background_sigs: Sequence[str] | None = ("SBS1", "SBS5")
    connected_sigs: bool | Sequence[Sequence[str]] = True
    activity_scale: ActivityScale = "burden"

    def __post_init__(self) -> None:
        if self.metric not in ("l2", "cosine"):
            raise ValueError(f"metric must be 'l2' or 'cosine', got {self.metric!r}.")
        if self.activity_scale not in ("burden", "raw"):
            raise ValueError(
                "activity_scale must be 'burden' or 'raw', got "
                f"{self.activity_scale!r}."
            )


#: Either refit strategy. Accepted by ``fit_signatures(strategy=...)``.
Strategy = Forward | Spa

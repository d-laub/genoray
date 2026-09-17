"""SigProfilerAssignment's ``cosmic_fit``, reimplemented.

Backward elimination from the saturated signature set, then add-remove
refinement layers. Transcribed from SigProfilerAssignment ``main``:
``single_sample.py`` (``fit_signatures``, ``add_signatures``,
``remove_all_single_signatures``, ``add_remove_signatures``,
``add_connected_sigs``) and ``decompose_subroutines.py``
(``process_sample``), ``solver="nnls"`` and ``pcawg_rule=False`` only.

Like SPA, the state carried between stages is a full-length exposure
*vector*, not an index set, and the active set is read off its nonzeros.
This matters: SPA's ``add_signatures`` rounds what it records, so a signature
whose rescaled activity falls below 0.5 drops out of the support without any
threshold testing it. Its removal sweep instead rounds only what it *reads*
and records the winning exposure unrounded; see :func:`_remove_all_single`.
Distances are always computed from the unrounded NNLS reconstruction.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._common import _cosine, _distance, _nnls, _round_conserve_sum

if TYPE_CHECKING:  # pragma: no cover
    from typing import Literal

    from ._strategy import Spa


def _support(h: NDArray[np.floating]) -> list[int]:
    """The active signature indices: the nonzeros of an exposure vector."""
    return [int(i) for i in np.nonzero(h)[0]]


def _exposure(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    active: Sequence[int],
    *,
    scale: Literal["burden", "burden-unrounded", "raw"],
) -> NDArray[np.float64]:
    """NNLS over ``active``, scattered into a full-length exposure vector.

    With ``scale="burden"`` the weights are renormalized to sum to the
    sample's mutation count and integer-rounded conserving that sum, which is
    how SPA reports its saturated fit, its removal-sweep inputs and exits, and
    its final exposure. Its add path rounds differently -- ``np.round`` plus a
    max-element repair (``single_sample.py:314-322``, ``:392-397``) -- which
    can move a mutation between signatures. With
    ``scale="burden-unrounded"`` the renormalization happens but the rounding
    does not, which is the exposure SPA records from its removal sweep (its
    ``np.round`` is commented out at ``single_sample.py:627``). With
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
    if scale != "raw":
        weights = weights / total * float(np.sum(m))
        if scale == "burden":
            weights = _round_conserve_sum(weights)
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


def _protected_positions(
    h: NDArray[np.floating], protected: Collection[int]
) -> list[int]:
    """SPA's ``get_changed_background_sig_idx``, transcribed.

    SPA's removal sweep works in the *compacted* index space of the exposure
    vector's nonzeros, so before each pass it remaps its protected set into
    that space and drops anything whose exposure is zero. That much is
    deliberate: a signature NNLS has already zeroed cannot be removed anyway.

    What is not deliberate is that SPA feeds the *compacted* result back in as
    if it were a full-length index on the next call. A protected signature
    therefore keeps its protection only while its compacted position happens
    to equal its full index -- that is, only while every lower-indexed
    signature is also active. In practice the first column of the reference
    (SBS1, for COSMIC) survives and the rest do not. See the sweep below and
    "Divergences from SPA" in the design note; the calibration test against
    real ``cosmic_fit`` is what forced this to be copied rather than cleaned
    up.

    SPA matches by exposure *value* rather than by index, which differs from
    this only when two active signatures carry the same exposure. There its
    behaviour is arbitrary, so it is not reproduced.
    """
    rank = {int(i): k for k, i in enumerate(np.nonzero(h)[0])}
    return [rank[i] for i in protected if i in rank]


def _remove_all_single(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    h: NDArray[np.floating],
    *,
    cutoff: float,
    metric: str,
    protected: Collection[int],
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

    ``protected`` is *not* an absolute veto. It is SPA's ``background_sigs``,
    and SPA's own handling of it decays across the sweep -- see
    :func:`_protected_positions`. The remap is recomputed at the two points
    SPA recomputes it: at the top of every pass against the exposure vector
    the sweep *started* from, and again against the winning vector after every
    accepted removal.

    Returns SPA's ``oldExposures`` -- the input rounded with the total
    conserved -- when no removal is ever accepted: every candidate protected,
    the best degradation above ``cutoff``, or one signature left at entry. As
    soon as a removal is accepted it returns instead that pass's winning
    exposure vector, rescaled to the sample's mutation count and left
    unrounded, because SPA records ``normalised_weights * sum(genomes)`` with
    its ``np.round`` commented out (``single_sample.py:610-612``, ``:627``)
    and only falls back to the rounded input when no removal was ever recorded
    (``:694-695``). Rounding here would drop low-activity signatures SPA
    keeps, since callers read the support off this vector's nonzeros.
    """
    h = np.asarray(h, dtype=np.float64)
    active = _support(h)
    if len(active) <= 1:
        return _round_conserve_sum(h)

    base = _distance(m, _reconstruction(W, m, active), metric)
    positions = list(protected)
    recorded: NDArray[np.float64] | None = None

    while len(active) > 1:
        # SPA remaps against the vector the sweep started from, never the
        # current one, so this is `h` on every pass.
        positions = _protected_positions(h, positions)
        best_d = np.inf
        best_h: NDArray[np.float64] | None = None
        for pos, i in enumerate(active):
            if pos in positions:
                continue
            cand = [j for j in active if j != i]
            d = _distance(m, _reconstruction(W, m, cand), metric)
            if d < best_d:
                best_d = d
                # Raw weights: SPA carries the unrounded winner (its np.round
                # is commented out, single_sample.py:627), and the raw NNLS
                # weights are a positive rescale of that vector. Both reads of
                # this vector -- _support and _protected_positions -- test only
                # zero-ness, which the raw weights preserve. scale="burden"
                # would not: _round_conserve_sum can zero entries.
                best_h = _exposure(W, m, cand, scale="raw")
        if best_h is None:
            break  # every remaining signature is protected
        if best_d - base > cutoff:
            break
        # SPA carries the winning exposure *vector*, so a signature NNLS gave
        # zero weight leaves the active set here without being removed.
        active = _support(best_h)
        base = best_d
        positions = _protected_positions(best_h, positions)
        recorded = _exposure(W, m, active, scale="burden-unrounded")

    return recorded if recorded is not None else _round_conserve_sum(h)


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
    """SPA's ``add_connected_sigs``: if any group member is active, add them all.

    Each group is tested against the *original* active set, not against the
    set being built. Upstream does the same, and it is what makes the result
    independent of the order ``groups`` happens to be listed in: expanding
    against the growing set would let ``((1,2),(2,3))`` reach 3 from 1 while
    ``((2,3),(1,2))`` would not.
    """
    original = frozenset(active)
    out = set(active)
    for group in groups:
        if original.intersection(group):
            out.update(group)
    return sorted(out)


def _fit_one_spa(
    W: NDArray[np.floating],
    m: NDArray[np.floating],
    spec: "Spa",
    *,
    protected: Collection[int],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], float]:
    """Refit one sample by SPA's ``cosmic_fit``.

    Four stages, matching ``process_sample`` then ``add_remove_signatures``:
    saturate over every signature, prune backward, refine with add-remove
    layers, then report on the requested activity scale.

    Returns ``(activities, cosine)``, the same contract as the forward path.
    """
    n_sigs = W.shape[1]
    empty = np.zeros(n_sigs, dtype=np.float64)
    if float(np.sum(m)) == 0.0:
        return empty, 0.0

    metric = spec.metric

    # Stage 1: saturate. SPA seeds every signature nonzero via a random dummy
    # exposure matrix; the draw's only effect is that every entry is nonzero.
    h = _exposure(W, m, range(n_sigs), scale="burden")
    if not _support(h):
        return empty, 0.0

    # Stage 2: initial prune. SPA passes background_sigs=[] here, so the
    # background signatures are NOT protected at this stage.
    h = _remove_all_single(
        W,
        m,
        h,
        cutoff=spec.initial_remove_penalty,
        metric=metric,
        protected=frozenset(),
    )

    # Stage 3: add-remove refinement layers.
    active = _expand_connected(sorted(set(_support(h)) | set(protected)), groups)
    best_d = np.inf
    best_active = active
    while True:
        present = _expand_connected(sorted(set(active) | set(protected)), groups)
        layer_d = np.inf
        layer_active: list[int] | None = None
        for cand in range(n_sigs):
            if cand in present:
                continue
            added = _try_add(
                W, m, present, cand, cutoff=spec.add_penalty, metric=metric
            )
            h_add = _exposure(W, m, added, scale="burden")
            h_rem = _remove_all_single(
                W,
                m,
                h_add,
                cutoff=spec.remove_penalty,
                metric=metric,
                protected=protected,
            )
            # SPA compares the add and remove supports here with a convoluted
            # expression that is not load-bearing: the removal sweep can only
            # shrink the support, so equal shapes already imply equal
            # supports, and equal supports imply equal distances. Taking the
            # post-removal support directly is exactly equivalent.
            pick = _support(h_rem)
            d = _distance(m, _reconstruction(W, m, pick), metric)
            if d < layer_d:
                layer_d = d
                layer_active = pick
        if layer_active is None or layer_d >= best_d:
            break
        best_d = layer_d
        best_active = layer_active
        active = layer_active

    # Stage 4: report.
    h = _exposure(W, m, best_active, scale=spec.activity_scale)
    cos = _cosine(m, W @ h) if _support(h) else 0.0
    return h, cos

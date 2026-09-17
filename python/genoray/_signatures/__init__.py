"""COSMIC mutational-signature refitting.

Decomposes a mutation catalogue into per-sample activities against a set of
reference signatures. Pure numpy/scipy/polars; no SigProfiler dependency.

Two strategies, selected via ``fit_signatures(strategy=...)``:

``Forward`` (the default) is genoray's own greedy forward selection from the
empty set, scored on cosine similarity, with a ``min_activity`` prune.

``Spa`` is a faithful reimplementation of SigProfilerAssignment's
``cosmic_fit``: one NNLS over the entire signature set, backward elimination
on relative L2 error, add-remove refinement layers, SBS1/SBS5 force-added as
background signatures on every layer (their protection decays within a removal
sweep, so they are not vetoes), force-added co-occurring partners, and
activities rescaled and integer-rounded to sum to the sample's mutation
burden. It recovers more true signatures than ``Forward`` at every burden
measured. ``tests/test_signatures_calibration.py`` checks it against the real
tool.

Note that ``Forward(criterion="cosine")`` and ``Spa`` are both
scale-invariant, and therefore blind to mutation burden: neither gains power
to resolve a real signature as the catalogue grows. Only
``Forward(criterion="bic")`` is a consistent estimator. Choosing SPA's search
direction improves recovery; it does not fix that.
"""

from __future__ import annotations

from ._common import _cosine as _cosine
from ._common import _nnls as _nnls
from ._common import _poisson_ll as _poisson_ll
from ._cosmic import _COSMIC_REGISTRY as _COSMIC_REGISTRY
from ._cosmic import _KIND_TOKEN as _KIND_TOKEN
from ._cosmic import _load_signature_file as _load_signature_file
from ._cosmic import cosmic_signatures
from ._fit import fit_signatures
from ._forward import Criterion
from ._forward import _fit_one as _fit_one
from ._strategy import SPA_CONNECTED_GROUPS as SPA_CONNECTED_GROUPS
from ._strategy import (
    ActivityScale,
    Forward,
    Metric,
    Spa,
    Strategy,
)

__all__ = [
    "SPA_CONNECTED_GROUPS",
    "ActivityScale",
    "Criterion",
    "Forward",
    "Metric",
    "Spa",
    "Strategy",
    "cosmic_signatures",
    "fit_signatures",
]

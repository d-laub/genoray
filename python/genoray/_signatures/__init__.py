"""COSMIC mutational-signature refitting.

A sparse forward-selection refit that decomposes a mutation catalogue into
per-sample activities against a set of reference signatures. Pure
numpy/scipy/polars; no SigProfiler dependency.

The *shape* follows SigProfilerAssignment's ``add_signatures`` -- greedily add
the signature that most improves the fit, stop once the improvement is too
small, then prune negligible activities -- but this is not a port, and the
difference is larger than the defaults. SigProfilerAssignment's ``cosmic_fit``
does not select forward at all: it runs one NNLS over the *entire* signature set
and then eliminates backward on relative L2 error. It also scores on relative L2
rather than cosine, force-includes SBS1/SBS5 as protected background signatures,
force-adds known co-occurring partners (``connected_sigs=True``), and rescales
and integer-rounds activities so they sum to the sample's total burden. None of
that is reproduced here. See the audit issue for the full comparison.

See ``fit_signatures``' ``criterion`` argument for the choice of stop rule, and
why the default is not the statistically consistent one.
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

__all__ = [
    "Criterion",
    "cosmic_signatures",
    "fit_signatures",
]

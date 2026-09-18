"""The shared "argument not passed" sentinel for the signature refit API.

Lives outside ``genoray._signatures`` so the cheap readers (``SparseVar``,
``SparseVar2``) can import it at module scope without pulling in scipy/joblib.
One object, imported by every producer and consumer, is what lets
``fit_signatures`` detect an explicitly-passed flat argument by identity.
"""

from __future__ import annotations

from typing import Any

#: Identity-checked sentinel distinguishing "argument not passed" from
#: "argument passed its default value". Needed so `strategy=Spa(), max_delta=0.01`
#: is still a conflict rather than silently accepted.
_UNSET: Any = object()

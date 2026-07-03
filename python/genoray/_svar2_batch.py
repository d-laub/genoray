"""M6b: raw two-channel batch-query methods for :class:`SparseVar2`.

Owned by the ``svar-2-m6b`` worktree. Mixed into ``SparseVar2`` so M6b and M6c
extend the class without both editing ``_svar2.py``.
"""

from __future__ import annotations


class _BatchQueryMixin:
    """Raw ``BatchResult`` → numpy query methods. Filled in M6b."""

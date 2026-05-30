"""Shared assertions for *_with_length tests.

The feature's guarantee: each haplotype carries enough length to cover the
original query span. Dense backends (VCF/PGEN) extend to one shared boundary
(the worst-case haplotype reaches Q); sparse (SparseVar) extends each
haplotype independently. ``clamped=True`` exempts cases where extension hit the
contig end and legitimately could not reach Q.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from genoray._utils import hap_ilens


def realized_hap_lengths(
    genos: NDArray[np.integer], ilens: NDArray[np.int32], q_len: int
) -> NDArray[np.int32]:
    """Realized haplotype lengths for a dense window. genos: (s p v)."""
    # base query length + net indel contribution carried by each haplotype
    return q_len + hap_ilens(genos, ilens)


def assert_dense_reaches_length(
    genos: NDArray[np.integer],
    ilens: NDArray[np.int32],
    q_len: int,
    *,
    clamped: bool = False,
) -> None:
    """Dense (VCF/PGEN): the worst-case haplotype must reach q_len unless clamped."""
    hap_lens = realized_hap_lengths(genos, ilens, q_len)
    if clamped:
        return
    assert hap_lens.max() >= q_len, (
        f"no haplotype reached query length {q_len}; max realized {hap_lens.max()}"
    )


def assert_sparse_reaches_length(
    hap_lens: NDArray[np.int32], q_len: int, *, clamped: bool = False
) -> None:
    """Sparse (SparseVar): every haplotype must independently reach q_len."""
    if clamped:
        return
    assert (hap_lens >= q_len).all(), (
        f"some haplotype did not reach query length {q_len}: {hap_lens}"
    )

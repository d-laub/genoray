"""Shared assertions for *_with_length tests.

The feature's guarantee (see memory ``with-length-semantics``): pick the most
PARSIMONIOUS set of ALT calls such that the personalized haplotype, padded on the
right with reference nucleotides, reaches the query length. Haplotype length
counts personalized nucleotides, not reference span.

Equivalent, checkable form per haplotype: let ``S`` be the selected ALT calls and
``R = q_len - sum(ilens[S])`` the reference footprint the haplotype must cover
(since hap_len = ref_bases_covered + sum_ilens, and we want hap_len == q_len, so
ref_bases_covered = R). The haplotype then spans reference ``[q_start, q_start+R)``
and is padded with plain reference where there are no ALT calls. The selection is
correct iff:

- **completeness:** every ALT call the haplotype carries with start in
  ``[q_start, q_start + R)`` is in ``S`` (none skipped inside the footprint), and
- **parsimony:** every selected call lies within the footprint (none included
  past where reference padding already reaches q_len).

Together these mean ``S`` is EXACTLY the carried ALT calls whose start falls in
``[q_start, q_start + R)``. The contig-end clamp needs no special case: if no
carried variants exist past the footprint, completeness holds trivially.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def assert_parsimonious_with_length(
    selected_per_hap: list[NDArray[np.integer]],
    carried_per_hap: list[NDArray[np.integer]],
    q_start: int,
    q_len: int,
    v_starts: NDArray[np.integer],
    ilens: NDArray[np.integer],
    *,
    backend: str = "",
) -> None:
    """Assert each haplotype's selected ALT calls are exactly the carried calls
    within the required reference footprint (see module docstring).

    Parameters
    ----------
    selected_per_hap
        Per-haplotype arrays of GLOBAL variant indices returned by a
        ``with_length`` read (the parsimonious selection).
    carried_per_hap
        Per-haplotype arrays of ALL GLOBAL variant indices the haplotype carries
        on the contig (from a plain, non-extended read).
    q_start, q_len
        0-based query start and its length (``end - start``).
    v_starts
        GLOBAL 0-based variant start positions, indexed by variant index.
    ilens
        GLOBAL ILEN per variant, indexed by variant index.
    backend
        Optional label for assertion messages.
    """
    assert len(selected_per_hap) == len(carried_per_hap)
    for h, (sel, carried) in enumerate(zip(selected_per_hap, carried_per_hap)):
        sel = np.asarray(sel)
        carried = np.asarray(carried)
        # selected must be a subset of what the haplotype actually carries
        assert set(sel.tolist()) <= set(carried.tolist()), (
            f"[{backend}] hap {h}: selected variants not all carried"
        )
        sum_ilen = int(ilens[sel].sum()) if len(sel) else 0
        footprint_end = q_start + (q_len - sum_ilen)
        in_footprint = carried[
            (v_starts[carried] >= q_start) & (v_starts[carried] < footprint_end)
        ]
        assert set(sel.tolist()) == set(in_footprint.tolist()), (
            f"[{backend}] hap {h}: selection is not the parsimonious footprint set. "
            f"q_start={q_start} q_len={q_len} sum_ilen={sum_ilen} "
            f"footprint=[{q_start},{footprint_end}); "
            f"selected={sorted(sel.tolist())} "
            f"expected={sorted(in_footprint.tolist())}"
        )
